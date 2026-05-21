# parallel_demo.py
import asyncio
import core.tools
from core.multi_agent.orchestrator import ResearchOrchestrator
from core.multi_agent.models import AgentTask, AgentRole, TaskStatus, ResearchPlan
from core.multi_agent.context import SharedContext
from core.multi_agent.queue import TaskQueue
from core.multi_agent.scheduler import ParallelScheduler
from core.multi_agent.agents import ResearcherAgent, WriterAgent, ReviewerAgent
from core.multi_agent.aggregator import ResultAggregator
from core.client import LLMClient


# ── 演示1：正常的并行执行 ─────────────────────────────────────────

def demo_parallel_research():
    print("=" * 60)
    print("演示1：并行研究（两个子课题同时进行）")
    print("=" * 60)

    orchestrator = ResearchOrchestrator()
    result = orchestrator.run(
        topic="Python 异步编程",
        goal="写一篇涵盖 asyncio 基础和高级用法的技术文章，含代码示例",
        verbose=True,
    )

    from pathlib import Path
    out = Path("workspace/parallel_report.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(result["report"], encoding="utf-8")
    print(f"\n✅ 报告已保存：{out}")


# ── 演示2：Sub-agent 失败时的容错行为 ────────────────────────────

class FailingResearcherAgent(ResearcherAgent):
    """故意失败的研究员，用于测试容错"""
    async def run_async(self, task, context):
        raise RuntimeError("模拟：搜索 API 连接超时")


async def demo_fault_tolerance():
    print("\n" + "=" * 60)
    print("演示2：容错——一个研究员失败，整体如何响应")
    print("=" * 60)

    # 构建一个有两个研究任务的计划
    t_r1 = AgentTask(
        id="r1", role=AgentRole.RESEARCHER,
        instruction="研究 asyncio 事件循环原理",
        depends_on=[],
    )
    t_r2 = AgentTask(
        id="r2", role=AgentRole.RESEARCHER,
        instruction="研究 asyncio 并发模式（这个会失败）",
        depends_on=[],
    )
    t_write = AgentTask(
        id="w1", role=AgentRole.WRITER,
        instruction="基于研究结果写文章",
        depends_on=["r1", "r2"],   # 依赖两个研究任务
    )
    t_review = AgentTask(
        id="rev1", role=AgentRole.REVIEWER,
        instruction="审核文章",
        depends_on=["w1"],
    )

    plan = ResearchPlan(topic="asyncio 并发", tasks=[t_r1, t_r2, t_write, t_review])
    context = SharedContext(topic="asyncio 并发")
    queue = TaskQueue(plan)

    # 用失败的研究员替换 r2
    agents = {
        AgentRole.RESEARCHER: ResearcherAgent(),
        AgentRole.WRITER:     WriterAgent(),
        AgentRole.REVIEWER:   ReviewerAgent(),
    }

    # 重写 r2 的执行逻辑：用 FailingResearcherAgent
    failing_researcher = FailingResearcherAgent()

    async def patched_execute(task, q, ctx, verbose):
        """r2 用失败的 Agent，r1 用正常的"""
        if task.id == "r2":
            import asyncio as _asyncio
            from core.multi_agent.scheduler import TASK_TIMEOUT_SECONDS
            q.mark_running(task.id)
            try:
                await _asyncio.wait_for(
                    failing_researcher.run_async(task, ctx),
                    timeout=TASK_TIMEOUT_SECONDS,
                )
            except Exception as e:
                q.mark_failed(task.id, str(e))
                print(f"  ❌ [researcher-r2] 失败：{e}")
        else:
            scheduler = ParallelScheduler(agents)
            await scheduler._execute_single(task, q, ctx, verbose)

    # 手动运行调度循环，观察级联效果
    print("\n任务计划：")
    print("  r1（研究员）→ 正常执行")
    print("  r2（研究员）→ 故意失败")
    print("  w1（写作者）→ 依赖 r1 和 r2，r2 失败后应被跳过")
    print("  rev1（审核）→ 依赖 w1，w1 跳过后应被跳过\n")

    # 并行执行 r1 和 r2
    print("第1轮（并行执行 r1 和 r2）：")
    await asyncio.gather(
        patched_execute(t_r1, queue, context, True),
        patched_execute(t_r2, queue, context, True),
    )

    # 观察级联跳过效果
    print(f"\n第2轮（检查 w1 状态）：w1 状态 = {t_write.status.value}")
    print(f"第3轮（检查 rev1 状态）：rev1 状态 = {t_review.status.value}")
    print(f"\n{queue.status_report()}")

    # 验证级联跳过
    assert t_write.status == TaskStatus.SKIPPED, "w1 应该被跳过"
    assert t_review.status == TaskStatus.SKIPPED, "rev1 应该被跳过"
    print("\n✅ 验证通过：r2 失败 → w1 被跳过 → rev1 被跳过（级联传播正确）")


# ── 演示3：部分成功时的降级输出 ──────────────────────────────────

async def demo_partial_success():
    print("\n" + "=" * 60)
    print("演示3：部分成功——两个研究员，一个成功一个失败")
    print("（写作者只用成功的研究结果）")
    print("=" * 60)

    # 两个研究任务彼此独立，写作任务只依赖其中一个
    t_r1 = AgentTask(
        id="r1", role=AgentRole.RESEARCHER,
        instruction="研究 asyncio 的 gather 和 wait 函数的用法",
        depends_on=[],
    )
    t_r2 = AgentTask(
        id="r2", role=AgentRole.RESEARCHER,
        instruction="研究 asyncio 的 Queue 和 Semaphore",
        depends_on=[],
    )
    # 写作只依赖 r1，r2 的成果是额外补充
    t_write = AgentTask(
        id="w1", role=AgentRole.WRITER,
        instruction="基于研究结果写一篇关于 asyncio 并发原语的文章",
        depends_on=["r1"],   # 只依赖 r1，r2 失败不影响写作
    )

    plan = ResearchPlan(
        topic="asyncio 并发原语",
        tasks=[t_r1, t_r2, t_write]
    )
    context = SharedContext(topic="asyncio 并发原语")
    queue = TaskQueue(plan)

    agents = {
        AgentRole.RESEARCHER: ResearcherAgent(),
        AgentRole.WRITER:     WriterAgent(),
        AgentRole.REVIEWER:   ReviewerAgent(),
    }
    scheduler = ParallelScheduler(agents)

    print("说明：r2 会正常运行，但 w1 只依赖 r1")
    print("即使 r2 完成，w1 也会在 r1 完成后立即开始\n")

    await scheduler.execute(plan, queue, context, verbose=True)

    # 聚合结果
    llm = LLMClient()
    aggregator = ResultAggregator(llm)
    report = aggregator.aggregate(plan, context)
    print(f"\n📄 报告长度：{len(report)} 字符")
    print(f"✅ 部分成功场景处理完毕")


if __name__ == "__main__":
    # 演示1：正常并行
    demo_parallel_research()

    # 演示2：级联失败
    asyncio.run(demo_fault_tolerance())

    # 演示3：部分成功
    asyncio.run(demo_partial_success())