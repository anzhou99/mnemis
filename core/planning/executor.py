# core/planning/executor.py
import asyncio
from datetime import datetime
from core.client import LLMClient
from core.models import Message
from core.planning.models import ExecutionPlan, PlanStep, StepStatus
from core.planning.planner import GoalPlanner
from core.tools.registry import get_registry
from core.planning.reflection import SelfCritic, ReflectionMemoryWriter
from utils.logger import get_logger

logger = get_logger(__name__)

STEP_EXECUTOR_SYSTEM = """你是一个专注执行单个步骤的 AI 助手。
你会收到一个具体的步骤描述和完成标准，以及已完成步骤的结果作为上下文。
你的任务是用可用的工具完成这一步，并在完成后用一段话总结你做了什么、结果是什么。
不要超出当前步骤的范围，专注完成好这一步就够了。"""


class PlanExecutor:
    """
    计划执行器：逐步执行 ExecutionPlan 里的每个步骤。
    支持：并行执行无依赖步骤 / 动态重规划 / 步骤间上下文传递
    """

    def __init__(
        self,
        planner: GoalPlanner,
        llm: LLMClient | None = None,
        max_step_retries: int = 2,
        verbose: bool = True,
        enable_reflection: bool = True,  # ← 新增：是否开启反思
        write_to_memory: bool = True,  # ← 新增：是否把经验写入记忆
        evolution_loop=None,  # ← 新增：可选的演进循环引用
    ):
        self.planner = planner
        self.llm = llm or LLMClient()
        self.max_step_retries = max_step_retries
        self.verbose = verbose
        self.enable_reflection = enable_reflection
        self.critic = SelfCritic(self.llm) if enable_reflection else None
        self.memory_writer = ReflectionMemoryWriter() if write_to_memory else None
        self.evolution_loop = evolution_loop

    async def execute(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        执行完整计划。
        自动识别可并行的步骤，处理失败时触发重规划。
        """
        if self.verbose:
            self._print_plan(plan)

        while not plan.is_complete():
            ready = plan.get_ready_steps()

            if not ready:
                # 检查是否有失败步骤导致其他步骤无法开始
                failed = plan.get_failed_steps()
                if failed:
                    logger.warning(f"{len(failed)} steps failed, triggering replan")
                    failure_analysis = self._analyze_failures(failed)
                    plan = self.planner.replan(plan, failure_analysis)
                    if not plan.get_ready_steps():
                        # 重规划后仍无可执行步骤，放弃
                        break
                else:
                    # 没有失败，也没有 ready 步骤——可能有循环依赖
                    logger.error("Deadlock detected in plan execution")
                    break
                continue

            if self.verbose and len(ready) > 1:
                print(f"\n⚡ 并行执行 {len(ready)} 个步骤：{[s.id for s in ready]}")

            # 并行执行所有 ready 步骤
            await asyncio.gather(*[self._execute_step(step, plan) for step in ready])

        # 生成最终总结
        plan.final_summary = self._generate_summary(plan)
        return plan

    # 升级 _execute_step，在成功后加入反思：
    async def _execute_step(self, step: PlanStep, plan: ExecutionPlan):
        for attempt in range(1, self.max_step_retries + 1):
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now()
            step.attempts = attempt

            if self.verbose:
                effort_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                    step.estimated_effort, "⚪"
                )
                print(f"\n{effort_icon} [{step.id}] {step.description[:60]}...")

            try:
                result = await self._run_step_with_tools(step, plan)
                step.result = result
                step.finished_at = datetime.now()
                plan.context[step.id] = result

                # ── 自我反思，加入质量上报 ──────────────────────────────────────
                if self.critic and self.evolution_loop:
                    reflection = self.critic.reflect(step)
                     # 把评分上报给演进监控器
                    self.evolution_loop.record_step_quality(
                        score=reflection.score,
                        step_description=step.description,
                    )

                    if self.verbose:
                        score_bar = "█" * int(reflection.score) + "░" * (
                            10 - int(reflection.score)
                        )
                        print(f"   🪞 反思：{reflection.score:.1f}/10 [{score_bar}]")
                        if reflection.weaknesses:
                            print(f"      不足：{reflection.weaknesses[0]}")

                    # 分数低且未达到里程碑：重试
                    if reflection.should_retry and attempt < self.max_step_retries:
                        if self.verbose:
                            print(f"   ⚠️  质量不达标（{reflection.score}/10），重试...")
                        step.status = StepStatus.PENDING
                        # 把改进建议加入下次尝试的 prompt
                        plan.context[f"{step.id}_improvement"] = (
                            reflection.improvement_suggestion
                        )
                        continue  # 重试

                    # 理解失败：触发重规划而非简单重试
                    if reflection.needs_replan:
                        if self.verbose:
                            print(f"   🔄 检测到理解偏差，触发重规划...")
                        step.status = StepStatus.FAILED
                        step.finished_at = datetime.now()
                        return  # 让 execute() 的主循环处理重规划

                    # 提炼并写入经验
                    if self.memory_writer and reflection.reusable_insight:
                        experience = self.critic.extract_experience(step, reflection)
                        if experience:
                            self.memory_writer.write_experience(
                                experience_text=experience,
                                source_step_id=step.id,
                            )
                            if self.verbose:
                                print(f"   💾 经验已记录：{experience[:60]}...")

                step.status = StepStatus.DONE
                if self.verbose:
                    print(f"   ✓ 完成（{step.duration_seconds:.1f}s）")
                return

            except Exception as e:
                logger.error(f"Step {step.id} attempt {attempt} failed: {e}")
                if attempt == self.max_step_retries:
                    step.status = StepStatus.FAILED
                    step.finished_at = datetime.now()
                    if self.verbose:
                        print(f"   ❌ 失败：{str(e)[:80]}")

    async def _run_step_with_tools(self, step: PlanStep, plan: ExecutionPlan) -> str:
        """
        用 LLM + 工具执行单个步骤。
        把已完成步骤的结果作为上下文传入。
        """
        # 构建上下文：当前步骤的依赖结果
        context_parts = []
        for dep_id in step.depends_on:
            dep_result = plan.context.get(dep_id, "")
            if dep_result:
                context_parts.append(f"[{dep_id} 的结果]\n{dep_result[:500]}")

        context_str = "\n\n".join(context_parts)
        prompt = f"""当前步骤：{step.description}

完成标准：{step.milestone}
"""
        if context_str:
            prompt = f"前置步骤的结果：\n{context_str}\n\n---\n\n{prompt}"

        # 用线程池执行同步 LLM 调用（支持并行）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_run_step,
            prompt,
        )
        return result

    def _sync_run_step(self, prompt: str) -> str:
        """同步执行一个步骤（在线程池里调用）"""
        registry = get_registry()
        tools = registry.get_schemas()

        messages = [{"role": "user", "content": prompt}]

        # 最多 3 轮工具调用
        for _ in range(3):
            msgs, tool_calls = self.llm.chat_with_tools(
                messages=messages,
                tools=tools,
                system=STEP_EXECUTOR_SYSTEM,
            )

            if not tool_calls:
                # 没有工具调用，提取最终文本
                return self._extract_text(msgs[-1])

            # 执行工具
            tool_results = []
            for tc in tool_calls:
                result = registry.execute(tc.name, tc.input)
                result.tool_use_id = tc.id
                tool_results.append(result)

            messages = list(msgs)
            messages = self.llm.append_tool_results(messages, tool_results)

        return self._extract_text(msgs[-1])

    def _analyze_failures(self, failed_steps: list[PlanStep]) -> str:
        """分析失败原因，为重规划提供参考"""
        analyses = []
        for s in failed_steps:
            analyses.append(f"步骤 [{s.id}] 失败：{s.description}")
        return "\n".join(analyses)

    def _generate_summary(self, plan: ExecutionPlan) -> str:
        """生成整个计划的执行总结"""
        done_steps = [s for s in plan.steps if s.status == StepStatus.DONE]
        if not done_steps:
            return "计划执行失败，没有成功完成任何步骤。"

        results_text = "\n".join(
            f"[{s.id}] {s.description}: {s.result[:200]}" for s in done_steps
        )

        response = self.llm.chat(
            messages=[
                Message(
                    role="user",
                    content=f"""请对以下目标的执行结果做一个简洁总结（300字以内）：

目标：{plan.goal}

各步骤完成情况：
{results_text}

总结执行成果，以及是否达成了原定目标。""",
                )
            ],
            system="你是执行总结专家，用简洁准确的语言总结任务完成情况。",
            temperature=0.3,
        )
        return response.content

    @staticmethod
    def _extract_text(message: dict) -> str:
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        return "".join(
            b.text if hasattr(b, "text") else b.get("text", "")
            for b in content
            if hasattr(b, "text") or (isinstance(b, dict) and b.get("type") == "text")
        )

    def _print_plan(self, plan: ExecutionPlan):
        print(f"\n🎯 目标：{plan.goal}")
        print(f"📋 计划：{len(plan.steps)} 个步骤\n")
        for step in plan.steps:
            deps = f" （依赖：{step.depends_on}）" if step.depends_on else ""
            effort = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                step.estimated_effort, "⚪"
            )
            print(f"  {effort} [{step.id}] {step.description}{deps}")
            print(f"      ✓ 完成标准：{step.milestone}")
