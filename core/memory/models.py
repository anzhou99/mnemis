from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


class MemoryType(str, Enum):
    PREFERENCE = "preference"  # 用户偏好：「喜欢简洁回答」
    FACT = "fact"  # 关于用户的事实：「是 Python 开发者」
    BACKGROUND = "background"  # 用户背景：「在做 AI Agent 项目」
    GOAL = "goal"  # 用户目标：「想开源并商业化」


@dataclass
class Memory:
    """一条语义记忆"""

    id: str
    content: str  # 记忆的自然语言描述
    memory_type: MemoryType
    confidence: float = 1.0  # 0.0-1.0
    source_session_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime | None = None
    access_count: int = 0
    vector_id: str | None = None

    def is_stale(self, days_threshold: int = 90) -> bool:
        """判断记忆是否已经很久未访问（代表可能已过时）"""
        if self.last_accessed is None:
            age = (datetime.now() - self.created_at).days
        else:
            age = (datetime.now() - self.last_accessed).days
        return age > days_threshold and self.access_count < 3


@dataclass
class Session:
    """一次对话会话"""

    id: str
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    title: str | None = None
    summary: str | None = None
    message_count: int = 0


@dataclass
class Message:
    """会话中的一条消息"""

    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    token_count: int | None = None


@dataclass
class MemoryConflict:
    """记忆冲突记录"""

    id: str
    new_memory_id: str
    old_memory_id: str
    conflict_type: Literal["contradiction", "update", "overlap"]
    resolution: Literal["new_wins", "old_wins", "both_kept"]
    resolved_at: datetime = field(default_factory=datetime.now)
