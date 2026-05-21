# 情景记忆管理器

from typing import Literal
import uuid
from datetime import datetime
from core.memory.database import MemoryDatabase
from core.memory.models import Session, Message
from core.client import LLMClient
from core.models import Message as LLMMessage
from utils.logger import get_logger

logger = get_logger(__name__)


# 触发压缩的消息条数阈值
COMPRESSION_THRESHOLD = 20
# 压缩时保留最近几条原文
KEEP_RECENT = 6


class EpisodicMemory:
    """
    情景记忆管理器。
    负责：对话存储 / 渐进式压缩 / 会话摘要生成 / 跨会话历史检索
    """

    def __init__(self, db: MemoryDatabase, llm_client: LLMClient) -> None:
        self.db = db
        self.llm = llm_client
        self._current_session: Session | None = None
        # 内存中维护当前会话的完整消息列表（用于渐进式压缩）
        self._messages_buffer: list[Message] = []
        # 已压缩的摘要列表（按时序）
        self._semmaries: list[str] = []

    # ── 会话管理 ──────────────────────────────────────────────────

    def start_session(self) -> Session:
        """开始一个新的对话会话"""
        self._current_session = self.db.create_session()
        self._messages_buffer = []
        self._summaries = []
        logger.info(f"Episodic moemory: session started {self._current_session.id}")
        return self._current_session

    def end_session(self) -> str | None:
        """
        结束当前会话：生成整体摘要，存入数据库。
        返回摘要文本。
        """
        if not self._current_session:
            return None

        all_messages = self.db.get_session_messages(session_id=self._current_session.id)
        if not all_messages:
            logger.debug(f"Session ended with no messages, shipping summary")
            return None

        logger.info(f"Ending session {self._current_session.id}, generating summary...")
        summary = self._generate_session_summary(all_messages)
        title = self._generate_title(summary)

        self.db.end_session(self._current_session.id, title, summary)
        logger.info(f"Session ended: '{title}'")

        self._current_session = None
        self._messages_buffer = []
        self._summaries = []
        return summary

    # ── 消息记录 ──────────────────────────────────────────────────
    def record_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        token_count: int | None = None,
    ):
        """记录一条消息，并在需要时触发渐进式压缩"""
        if not self._current_session:
            logger.warning("record_message called without active session")
            return

        msg = self.db.save_message(self._current_session.id, role, content, token_count)
        self._messages_buffer.append(msg)

        # 适时压缩
        if len(self._messages_buffer) >= COMPRESSION_THRESHOLD:
            self._compress_early_messages()

    def _compress_early_messages(self):
        """
        把 buffer 里较早的消息压缩成摘要，只保留最近 KEEP_RECENT 条原文。
        """
        to_compress = self._messages_buffer[:-KEEP_RECENT]
        recent = self._messages_buffer[-KEEP_RECENT:]
        if not to_compress:
            return

        logger.debug(f"Compressing {len(to_compress)} messages into summary")
        summary = self._summarize_messages(to_compress)
        self._semmaries.append(summary)
        self._messages_buffer = recent
        logger.debug(
            f"Compression done. Summaries: {len(self._summaries)}, Buffer: {len(self._messages_buffer)}"
        )

    # ── 构建传给 LLM 的 messages ──────────────────────────────────

    def build_context_messages(self) -> list[dict]:
        """
        把当前会话的上下文构建成传给 LLM 的 messages 格式。
        格式：[摘要（以 system 风格注入）] + [最近原文消息]
        """
        result = []

        # 把所有摘要合并成一条 user 消息注入到最前面
        if self._summaries:
            combined = "\n\n".join(
                f"[早期对话摘要 {i+1}]\n{s}" for i, s in enumerate(self._semmaries)
            )
            result.append(
                {
                    "role": "user",
                    "content": f"以下是本次会话早期对话的摘要，供参考：\n\n{combined}",
                }
            )
            result.append(
                {
                    "role": "assistant",
                    "content": "好的，我已了解早期对话内容，继续为您服务。",
                }
            )

        # 追加最近的原文消息
        for msg in self._messages_buffer:
            result.append({"role": msg.role, "content": msg.content})

        return result

    # ── 跨会话历史检索 ────────────────────────────────────────────
    def get_recent_history(self, limit: int = 5) -> str:
        """
        获取最近 N 次会话的摘要，用于新会话开始时的上下文注入。
        """
        sessions = self.db.get_recent_sessions(limit)
        if not sessions:
            return ""

        parts = []
        for s in sessions:
            if not s.summary:
                continue
            date_str = s.started_at.strftime("%Y-%m-%d")
            parts.append(f"[{date_str} {s.title or '未命名会话'} \n {s.summary}]")

        if not parts:
            return ""

        return "## 历史会话摘要（最近 {} 次）\n\n".format(len(parts)) + "\n\n".join(
            parts
        )

    # ── LLM 调用：生成摘要和标题 ──────────────────────────────────
    def _summarize_messages(self, messages: list[Message]) -> str:
        """用 LLM 把一批消息压缩成摘要"""
        dialogue = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        response = self.llm.chat(
            messages=[
                LLMMessage(
                    role="user",
                    content=f"""请把以下对话压缩成简洁的摘要（200字以内）。
                        保留关键信息：讨论了什么话题、得出了什么结论、用户有什么需求。

                        对话内容：
                        {dialogue}""",
                )
            ],
            system="你是对话摘要助手，擅长提炼对话要点。输出简洁的中文摘要，不超过200字。",
            temperature=0.3,
        )
        return response.content

    def _generate_session_summary(self, messages: list[Message]) -> str:
        """为整个会话生成摘要"""
        # 如果消息太多，只取首尾各 10 条
        if len(messages) > 20:
            sample = messages[:10] + messages[-10:]
            note = f"（共 {len(messages)} 条，展示首尾各10条）"
        else:
            sample = messages
            note = ""

        dialogue = "\n".join(f"{m.role.upper()}: {m.content[:400]}" for m in sample)

        response = self.llm.chat(
            messages=[
                LLMMessage(
                    role="user",
                    content=f"""请为以下完整会话生成摘要{note}。
                包含：主要讨论的话题、重要结论、用户的需求和偏好。限 300 字。

                {dialogue}""",
                )
            ],
            system="你是会话摘要专家。生成结构清晰、信息密度高的中文摘要。",
            temperature=0.3,
        )
        return response.content

    def _generate_title(self, summary: str) -> str:
        """从摘要生成 10 字以内的会话标题"""
        response = self.llm.chat(
            messages=[
                LLMMessage(
                    role="user",
                    content=f"根据以下摘要，生成一个 10 字以内的对话标题。只输出标题本身。\n\n{summary}",
                )
            ],
            system="你是标题生成助手。只输出标题，不超过10个字，不加任何解释。",
            temperature=0.1,
        )
        return response.content.strip().strip("「」《》【】")
