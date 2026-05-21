from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLANNING = "replanning"


class EffortLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PlanStep:
    """
    计划中的一个执行步骤。
    """

    id: str = field(default_factory=lambda: f"step-{uuid.uuid4().hex[:6]}")
    description: str = ""
    milestone: str = ""
    depends_on: list[str] = field(default_factory=list)
    estimated_effort: EffortLevel = EffortLevel.MEDIUM  # low/medium/high

    # 执行状态
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    relection: str = ""  # 执行后的自我反思
    quality_score: float | None = None  # 1-10 评分
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int = 0  # 尝试次数（用于重试控制）

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def is_ready(self, completed_ids: set[str]) -> bool:
        return all(dep in completed_ids for dep in self.depends_on)


@dataclass
class ExecutionPlan:
    """
    完整的执行计划。
    包含目标、步骤列表、执行历史和重规划记录。
    """

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)  # 步骤间共享的数据
    created_at: datetime = field(default_factory=datetime.now)
    replan_count: int = 0  # 重规划次数（防止无限重规划）
    final_summary: str = ""

    def get_ready_steps(self) -> list[PlanStep]:
        """返回所有依赖已满足、可以立即执行的步骤"""
        completed = {s.id for s in self.steps if s.status == StepStatus.DONE}
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING and s.is_ready(completed)
        ]

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED)
            for s in self.steps
        )

    def get_failed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def progress_summary(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        running = sum(1 for s in self.steps if s.status == StepStatus.RUNNING)
        return (
            f"{done}/{total} 完成"
            + (f", {running} 进行中" if running else "")
            + (f", {failed} 失败" if failed else "")
        )
