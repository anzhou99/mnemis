import asyncio
from core.multi_agent.models import AgentTask, TaskStatus, ResearchPlan
from utils.logger import get_logger

logger = get_logger(__name__)


class TaskQueue:
    """
    任务调度队列。
    Orchestrator 把任务放入队列，调度器按依赖关系决定执行顺序。

    支持：
    - 依赖关系感知（有依赖的任务等依赖完成后才执行）
    - 状态追踪（pending / running / done / failed / skipped）
    - 失败传播（依赖失败时跳过下游任务）
    """

    def __init__(self, plan: ResearchPlan):
        self.plan = plan
        # task_id → AgentTask 的快速查找
        self._task_map: dict[str, AgentTask] = {t.id: t for t in plan.tasks}

    # ── 任务状态管理 ──────────────────────────────────────────────
    def mark_running(self, task_id: str):
        task = self._task_map[task_id]
        task.status = TaskStatus.RUNNING
        from datetime import datetime

        task.started_at = datetime.now()
        logger.debug(f"Task {task_id} [{task.role.value}] → RUNNING")

    def mark_done(self, task_id: str, result: str):
        task = self._task_map[task_id]
        task.status = TaskStatus.DONE
        task.result = result
        from datetime import datetime

        task.finished_at = datetime.now()
        logger.info(
            f"Task {task_id} [{task.role.value}] → DONE "
            f"({task.duration_seconds:.1f}s, {len(result)} chars)"
        )

    def mark_failed(self, task_id: str, error: str):
        task = self._task_map[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        from datetime import datetime

        task.finished_at = datetime.now()
        logger.error(f"Task {task_id} [{task.role.value}] → FAILED: {error[:80]}")

        # 级联跳过：依赖这个失败任务的下游任务全部标记为 SKIPPED
        self._cascade_skip(task_id)

    def mark_skipped(self, task_id: str, error: str | None = None):
        task = self._task_map[task_id]
        task.status = TaskStatus.SKIPPED
        if error:
            task.error = error

        from datetime import datetime

        task.finished_at = datetime.now()
        logger.error(f"Task {task_id} [{task.role.value}] → SKIPPED. " f"{error}")

        # 级联跳过：依赖这个失败任务的下游任务全部标记为 SKIPPED
        self._cascade_skip(task_id)

    def _cascade_skip(self, failed_task_id: str):
        """把依赖失败任务的所有下游任务标记为 SKIPPED"""
        for task in self.plan.tasks:
            if failed_task_id in task.depends_on and task.status == TaskStatus.PENDING:
                self.mark_skipped(
                    task.id, f"Skipped: dependency {failed_task_id} failed"
                )

                self._cascade_skip(task.id)

    # ── 调度逻辑 ──────────────────────────────────────────────────
    def get_ready_tasks(self) -> list[AgentTask]:
        """获取所有依赖已满足、可以立即执行的任务"""
        completed_ids = {
            t.id
            for t in self.plan.tasks
            if t.status in (TaskStatus.DONE, TaskStatus.SKIPPED)
        }
        return [
            t
            for t in self.plan.tasks
            if t.status == TaskStatus.PENDING and t.is_ready(completed_ids)
        ]

    def has_pending(self) -> bool:
        """是否还有未完成的任务"""
        return any(
            t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            for t in self.plan.tasks
        )

    def is_all_done(self) -> bool:
        return all(
            t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in self.plan.tasks
        )

    # ── 状态报告 ──────────────────────────────────────────────────
    def status_report(self) -> str:
        """生成可读的任务状态报告"""
        lines = [f"任务队列状态（{self.plan.topic}）："]
        status_icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.DONE: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.SKIPPED: "⏭️",
        }
        for task in self.plan.tasks:
            icon = status_icons[task.status]
            duration = (
                f" ({task.duration_seconds:.1f}s)" if task.duration_seconds else ""
            )
            lines.append(
                f"  {icon} [{task.id}] {task.role.value}{duration}"
                + (f" → {task.error[:40]}" if task.error else "")
            )
        return "\n".join(lines)
