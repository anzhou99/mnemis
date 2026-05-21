import json
from core.client import LLMClient
from core.models import Message
from core.planning.models import ExecutionPlan, PlanStep, StepStatus
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_REPLAN_TIMES = 3  # 最多重规划次数，防止无限循环

PLANNER_SYSTEM = """你是一位专业的项目规划专家，擅长把模糊的高层目标分解为具体可执行的步骤。

规划原则：
1. 每个步骤必须有清晰的完成标准（milestone），能被客观验证
2. 正确识别步骤间的依赖关系——没有依赖的步骤可以并行执行
3. 步骤粒度适中：太粗（比如「完成整个项目」）和太细（比如「打开浏览器」）都不好
4. 每步估算工作量：low（<1小时）/ medium（1-4小时）/ high（>4小时）

只输出 JSON，不要任何其他内容。"""


PLAN_TEMPLATE = """请把以下目标分解为可执行的步骤计划：

目标：{goal}
背景信息：{context}

输出 JSON 格式：
{{
  "understanding": "用一句话说明你对这个目标的理解",
  "steps": [
    {{
      "id": "step-1",
      "description": "具体要做什么",
      "milestone": "完成时的可验证标准",
      "depends_on": [],
      "estimated_effort": "low|medium|high"
    }}
  ]
}}

注意：
- depends_on 填写前置步骤的 id，没有依赖时填空列表
- 步骤数量控制在 3-8 个，不要过度拆解"""

REPLAN_TEMPLATE = """执行计划中出现了问题，需要重新规划后续步骤。

原始目标：{goal}
已完成步骤：
{completed_steps}

失败/遇到问题的步骤：
{failed_steps}

原因分析：{failure_analysis}

请重新规划剩余步骤。只输出需要新增或修改的步骤（JSON 格式），保持已完成步骤不变：
{{
  "analysis": "对问题的分析和调整思路",
  "new_steps": [
    {{
      "id": "step-新id",
      "description": "调整后的步骤描述",
      "milestone": "完成标准",
      "depends_on": ["已完成步骤的id或新步骤id"],
      "estimated_effort": "low|medium|high"
    }}
  ]
}}"""


class GoalPlanner:
    """
    目标规划器：把高层目标分解为带依赖关系的执行步骤列表。
    支持初始规划和动态重规划。
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def plan(self, goal: str, context: dict | None = None) -> ExecutionPlan:
        """
        把目标分解为执行计划。
        context 是可选的背景信息，帮助生成更针对性的计划。
        """
        ctx_str = json.dumps(context or {}, ensure_ascii=False) if context else "无"
        logger.info(f"Planning goal: {goal}")

        response = self.llm.chat(
            messages=[
                Message(
                    role="user",
                    content=PLAN_TEMPLATE.format(goal=goal, context=ctx_str),
                )
            ],
            system=PLANNER_SYSTEM,
            temperature=0.3,
        )

        steps = self._parse_plan_response(response.content)
        plan = ExecutionPlan(goal=goal, steps=steps)

        logger.info(
            f"Plan created: {len(steps)} steps, "
            f"{sum(1 for s in steps if not s.depends_on)} can start immediately"
        )

        return plan

    def replan(self, plan: ExecutionPlan, failure_analysis: str) -> ExecutionPlan:
        """
        动态重规划：在执行中途遇到问题时，重新规划剩余步骤。
        保留已完成的步骤，只修改或新增后续步骤。
        """
        if plan.replan_count >= MAX_REPLAN_TIMES:
            logger.warning(f"Max replan count ({MAX_REPLAN_TIMES}) reached, stopping")
            return plan

        completed = [s for s in plan.steps if s.status == StepStatus.DONE]
        failed = [s for s in plan.steps if s.status == StepStatus.FAILED]

        completed_str = "\n".join(
            f"  [{s.id}] {s.description} → {s.result[:100]}" for s in completed
        )
        failed_str = "\n".join(f"  [{s.id}] {s.description}" for s in failed)

        response = self.llm.chat(
            messages=[
                Message(
                    role="user",
                    content=REPLAN_TEMPLATE.format(
                        goal=plan.goal,
                        completed_steps=completed_str or "（无）",
                        failed_steps=failed_str or "（无）",
                        failure_analysis=failure_analysis,
                    ),
                )
            ],
            system=PLANNER_SYSTEM,
            temperature=0.3,
        )

        new_steps = self._parse_replan_response(response.content)
        # 把失败步骤标记为 SKIPPED，追加新步骤
        for s in plan.steps:
            if s.status == StepStatus.FAILED:
                s.status = StepStatus.SKIPPED

        plan.steps.extend(new_steps)
        plan.replan_count += 1

        logger.info(f"Replan #{plan.replan_count}: added {len(new_steps)} new steps")
        return plan

    # ── 私有方法 ──────────────────────────────────────────────────

    def _parse_plan_response(self, content: str) -> list[PlanStep]:
        """解析 LLM 的计划输出"""
        try:
            raw = (
                content.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(raw)
            return [
                PlanStep(
                    id=s.get("id", f"step-{i+1}"),
                    description=s.get("description", ""),
                    milestone=s.get("milestone", ""),
                    depends_on=s.get("depends_on", []),
                    estimated_effort=s.get("estimated_effort", "medium"),
                )
                for i, s in enumerate(data.get("steps", []))
            ]
        except Exception as e:
            logger.error(f"Plan parsing failed: {e}")
            # 降级：返回单步计划
            return [
                PlanStep(
                    id="step-1",
                    description=f"完成目标：{content[:100]}",
                    milestone="任务完成",
                )
            ]

    def _parse_replan_response(self, content: str) -> list[PlanStep]:
        """解析重规划响应"""
        try:
            raw = (
                content.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            data = json.loads(raw)
            return [
                PlanStep(
                    id=s.get("id", f"replan-{i+1}"),
                    description=s.get("description", ""),
                    milestone=s.get("milestone", ""),
                    depends_on=s.get("depends_on", []),
                    estimated_effort=s.get("estimated_effort", "medium"),
                )
                for i, s in enumerate(data.get("new_steps", []))
            ]
        except Exception as e:
            logger.error(f"Replan parsing failed: {e}")
            return []
