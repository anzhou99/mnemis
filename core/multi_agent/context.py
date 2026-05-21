import threading
from datetime import datetime
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)


class SharedContext:
    """
    Multi-Agent 系统的共享状态容器。
    线程安全（为 P4·5 的并行执行做准备）。

    存储结构：
    - topic: 研究主题
    - task_results: {task_id: result_str}（各 Agent 的产出）
    - metadata: {key: value}（任意元数据）
    - messages: 通信日志
    """

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._lock = threading.Lock()
        self._task_results: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}
        self._messages: list[dict] = []
        self.created_at = datetime.now()

        logger.debug(f"SharedContext created for topic: {topic[:50]}")

    # ── 任务结果 ──────────────────────────────────────────────────
    def set_result(self, task_id: str, result: str, agent_role: str = ""):
        """Sub-agent 完成任务后，把结果写入共享上下文"""
        with self._lock:
            self._task_results[task_id] = result
            logger.debug(
                f"Context updated | task={task_id} | "
                f"agent={agent_role} | chars={len(result)}"
            )

    def get_result(self, task_id: str) -> str | None:
        """获取指定任务的结果"""
        with self._lock:
            return self._task_results.get(task_id)

    def get_results_by_role(self, role: str, tasks) -> list[str]:
        """
        获取指定角色的所有任务结果。
        tasks: AgentTask 列表，用于过滤。
        """
        with self._lock:
            return [
                self._task_results[t.id]
                for t in tasks
                if t.role.value == role and t.id in self._task_results
            ]

    # ── 元数据 ────────────────────────────────────────────────────

    def set(self, key: str, value: Any):
        """存储任意元数据"""
        with self._lock:
            self._metadata[key] = value

    def get(self, key: str, default: Any = None):
        """读取元数据"""
        with self._lock:
            return self._metadata.get(key, default)

    # ── 消息日志 ──────────────────────────────────────────────────
    def log_message(self, sender: str, receiver: str, content: str):
        """记录 Agent 间的通信日志，方便调试"""
        with self._lock:
            self._messages.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "sender": sender,
                    "receiver": receiver,
                    "content": content[:200],  # 截断，只存摘要
                }
            )

    # ── 状态快照 ──────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """返回当前状态的快照，用于调试和日志"""
        with self._lock:
            return {
                "topic": self.topic,
                "task_results_count": len(self._task_results),
                "task_ids": list(self._task_results.keys()),
                "metadata_keys": list(self._metadata.keys()),
                "message_count": len(self._messages),
            }
