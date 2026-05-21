import asyncio
import time
from core.multi_agent.models import AgentTask, AgentRole, TaskStatus, ResearchPlan
from core.multi_agent.context import SharedContext
from core.multi_agent.queue import TaskQueue
from utils.logger import get_logger

logger = get_logger(__name__)


# 单个 Sub-agent 的最长执行时间（秒）
TASK_TIMEOUT_SECONDS = 120


class ParallelScheduler:
    """
    并行任务调度器。

    核心算法：
    1. 扫描任务图，找出当前所有依赖已满足的任务（ready tasks）
    2. 并行执行所有 ready tasks（asyncio.gather）
    3. 任意任务完成后，重新扫描，启动新的 ready tasks
    4. 重复直到所有任务完成或全部失败/跳过
    """

    def __init__(self, agents: dict) -> None:
        self.agents = agents  # AgentRole → BaseSubAgent

    async def execute(
        self,
        plan: ResearchPlan,
        queue: TaskQueue,
        context: SharedContext,
        verbose: bool = True,
    ):
        """
        执行研究计划，自动识别可并行的任务。
        """
        start_time = time.time()

        while not queue.is_all_done():
            ready = queue.get_ready_tasks()

            if not ready:
                # 没有 ready 任务但还有 running 任务——等待中
                # （不应该发生在同步调用里，但并发场景下可能出现竞态）
                await asyncio.sleep(0.1)

                continue
            if verbose:
                roles = [t.role.value for t in ready]
                if len(ready) > 1:
                    print(f"\n⚡ 并行执行 {len(ready)} 个任务：{roles}")
                else:
                    emoji = {"researcher": "🔍", "writer": "✍️", "reviewer": "🔎"}
                    print(
                        f"\n{emoji.get(ready[0].role.value, '▶')} [{ready[0].role.value}] 开始..."
                    )

            # 并行执行所有 ready 任务
            await asyncio.gather(
                *[self._execute_single(task, queue, context, verbose) for task in ready]
            )

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n⏱ 总耗时：{elapsed:.1f}s")
            print(queue.status_report())

    async def _execute_single(
        self, task: AgentTask, queue: TaskQueue, context: SharedContext, verbose: bool
    ):
        """
        执行单个任务，包含：
        - 超时控制
        - 错误隔离（异常不传播到调用方）
        - 结果写入 SharedContext
        """
        agent = self.agents.get(task.role)
        if not agent:
            queue.mark_failed(task.id, f"未找到角色 {task.role.value} 对应的 Agent")
            return

        queue.mark_running(task.id)
        context.set(f"task_role_{task.id}", task.role.value)

        try:
            # 带超时的异步执行
            result = await asyncio.wait_for(
                agent.run_async(task, context), timeout=TASK_TIMEOUT_SECONDS
            )

            context.set_result(task.id, result, task.role.value)
            queue.mark_done(task.id, result)

            if verbose:
                print(f"  ✓ [{task.role.value}] 完成（{len(result)} 字符）")

        except asyncio.TimeoutError:
            error = f"执行超时（>{TASK_TIMEOUT_SECONDS}s）"
            logger.error(f"Task {task.id} timeout")
            queue.mark_failed(task.id, error)
            if verbose:
                print(f"  ⏰ [{task.role.value}] 超时！{error}")

        except Exception as e:
            error = str(e)
            logger.error(f"Task {task.id} failed: {error}", exc_info=True)
            queue.mark_failed(task.id, error)
            if verbose:
                print(f"  ❌ [{task.role.value}] 失败：{error[:80]}")
