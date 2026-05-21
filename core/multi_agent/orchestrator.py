import asyncio
import uuid
from core.client import LLMClient
from core.models import Message
from core.multi_agent.models import AgentTask, AgentRole, TaskStatus, ResearchPlan
from core.multi_agent.context import SharedContext
from core.multi_agent.queue import TaskQueue
from core.multi_agent.aggregator import ResultAggregator
from core.multi_agent.agents import ResearcherAgent, WriterAgent, ReviewerAgent
from core.multi_agent.scheduler import ParallelScheduler
from utils.logger import get_logger

logger = get_logger(__name__)

ORCHESTRATOR_SYSTEM = """你是研究项目协调者，负责把复杂研究任务分解为并行可执行的工作计划。
只输出 JSON，不要任何其他内容。"""

PLAN_PROMPT = """为以下研究任务制定执行计划：

主题：{topic}
目标：{goal}

请把研究阶段分解为 1-3 个独立子课题（可并行研究），然后汇总写作和审核。

输出 JSON：
{{
  "research_subtopics": [
    {{"id": "r1", "focus": "子课题1的具体研究方向", "keywords": ["关键词1", "关键词2"]}},
    {{"id": "r2", "focus": "子课题2的具体研究方向", "keywords": ["关键词3"]}}
  ],
  "write_instruction": "给写作者的详细指令：文章结构、字数、风格、代码示例要求",
  "review_instruction": "给审核者的详细指令：审核重点和评分标准"
}}

注意：research_subtopics 之间必须相互独立，可以同时搜索，不要有先后依赖。"""


class ResearchOrchestrator:
    """
    研究团队的协调者。
    负责：制定计划 → 分配任务 → 监控执行 → 聚合结果
    """

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.aggregator = ResultAggregator(self.llm)

        # 三个专门化 Sub-agent
        self._agents = {
            AgentRole.RESEARCHER: ResearcherAgent(),
            AgentRole.WRITER: WriterAgent(),
            AgentRole.REVIEWER: ReviewerAgent(),
        }

        self.scheduler = ParallelScheduler(self._agents)

    def run(
        self,
        topic: str,
        goal: str = "写一篇深度技术分析文章",
        verbose: bool = True,
    ) -> dict:
        """同步入口（内部用 asyncio 并行）"""
        return asyncio.run(self.run_async(topic, goal, verbose))

    async def run_async(
        self,
        topic: str,
        goal: str = "写一篇深度技术分析文章",
        verbose: bool = True,
    ) -> dict:
        """异步核心：支持并行 Sub-agent"""

        if verbose:
            print(f"\n🎯 研究主题：{topic}")
            print("📋 制定研究计划...")

        plan = self._make_plan(topic, goal)
        context = SharedContext(topic=topic)
        queue = TaskQueue(plan)

        if verbose:
            self._print_plan(plan)

        # 用并行调度器执行
        await self.scheduler.execute(plan, queue, context, verbose)

        if verbose:
            print("\n📊 聚合最终报告...")

        report = self.aggregator.aggregate(plan, context)
        plan.final_report = report
        plan.status = TaskStatus.DONE

        return {
            "report": report,
            "plan_summary": queue.status_report(),
            "context": context,
        }

    def _make_plan(self, topic: str, goal: str) -> ResearchPlan:
        """用 LLM 制定研究计划，生成任务列表"""
        import json

        response = self.llm.chat(
            messages=[
                Message(role="user", content=PLAN_PROMPT.format(topic=topic, goal=goal))
            ],
            system=ORCHESTRATOR_SYSTEM,
            temperature=0.2,
        )

        try:
            raw = (
                response.content.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            plan_data = json.loads(raw)
        except Exception as e:
            logger.warning(
                f"Plan parsing failed: {e}, using single-researcher fallback"
            )
            plan_data = {
                "research_subtopics": [
                    {"id": "r1", "focus": f"全面研究：{topic}", "keywords": [topic]}
                ],
                "write_instruction": f"基于研究结果，写一篇关于 {topic} 的深度文章，约 1500 字。",
                "review_instruction": "审核技术准确性、逻辑完整性、表达清晰度。",
            }

        tasks: list[AgentTask] = []
        research_ids: list[str] = []
        # 生成并行研究任务（每个子课题一个 ResearcherAgent）
        for sub in plan_data.get("research_subtopics", []):
            task_id = f"r_{sub['id']}"
            research_ids.append(task_id)
            keywords = "、".join(sub.get("keywords", []))
            tasks.append(
                AgentTask(
                    id=task_id,
                    role=AgentRole.RESEARCHER,
                    instruction=(
                        f"研究方向：{sub['focus']}\n"
                        f"关键词：{keywords}\n"
                        f"请深入搜索并整理这个方向的关键信息，"
                        f"用 Markdown 格式输出结构化研究笔记。"
                    ),
                    depends_on=[],  # 研究任务之间无依赖，可并行
                )
            )

        # 写作任务依赖所有研究任务
        write_id = f"w_{uuid.uuid4().hex[:6]}"
        tasks.append(
            AgentTask(
                id=write_id,
                role=AgentRole.WRITER,
                instruction=plan_data.get("write_instruction", ""),
                depends_on=research_ids,  # 等所有研究完成
            )
        )

        # 审核任务依赖写作任务
        rev_id = f"rev_{uuid.uuid4().hex[:6]}"
        tasks.append(
            AgentTask(
                id=rev_id,
                role=AgentRole.REVIEWER,
                instruction=plan_data.get("review_instruction", ""),
                depends_on=[write_id],
            )
        )

        return ResearchPlan(topic=topic, tasks=tasks)

    @staticmethod
    def _print_plan(plan: ResearchPlan):
        """打印任务计划，含并行关系"""
        researcher_tasks = [t for t in plan.tasks if t.role == AgentRole.RESEARCHER]
        other_tasks = [t for t in plan.tasks if t.role != AgentRole.RESEARCHER]

        print(f"✓ 计划：{len(plan.tasks)} 个任务")
        if len(researcher_tasks) > 1:
            print(f"  ⚡ 研究阶段：{len(researcher_tasks)} 个研究员并行")
        for t in researcher_tasks:
            print(f"    [{t.id}] {t.instruction[:60]}...")
        for t in other_tasks:
            deps = f"（等待：{t.depends_on}）"
            print(f"  [{t.id}] {t.role.value} {deps}")
