from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class TaskStatus(str, Enum):
    PENDING = "pending"  # 等待执行
    RUNNING = "running"  # 正在执行
    DONE = "done"  # 成功完成
    FAILED = "failed"  # 执行失败
    SKIPPED = "skipped"  # 被跳过（依赖失败时）


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"


@dataclass
class AgentTask:
    """
    分配给 Sub-agent 的一个任务单元。
    包含任务描述、依赖关系、执行结果。
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: AgentRole = AgentRole.RESEARCHER
    instruction: str = ""  # 给这个 Agent 的具体指令
    context: dict = field(default_factory=dict)  # 任务所需的上下文数据
    depends_on: list[str] = field(default_factory=list)  # 依赖的任务 ID 列表

    # 执行状态
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def is_ready(self, completed_task_ids: set[str]) -> bool:
        return all(dep in completed_task_ids for dep in self.depends_on)


@dataclass
class AgentMessage:
    """
    Agent 间传递的标准化消息。
    比裸字符串更可靠：携带发送方、接收方、类型信息。
    """

    sender: AgentRole
    receiver: AgentRole
    msg_type: str  # "task_result" / "request_clarification" / "status_update"
    content: str  # 消息正文
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchPlan:
    """
    Orchestrator 制定的研究计划。
    包含完整的任务列表和执行顺序。
    """

    topic: str
    tasks: list[AgentTask] = field(default_factory=list)
    final_report: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def get_ready_tasks(self) -> list[AgentTask]:
        """获取所有依赖已满足、可以立即执行的任务"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.DONE}

        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING and t.is_ready(completed_ids)
        ]

    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in self.tasks
        )

    def summary(self) -> str:
        def countStatus(status: TaskStatus):
            return sum(1 for t in self.tasks if t.status == status)

        done = countStatus(TaskStatus.DONE)
        failed = countStatus(TaskStatus.FAILED)
        total = len(self.tasks)

        return f"总计-{total}; 成功-{done}; 失败-{failed}"
