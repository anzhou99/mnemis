# core/memory/manager.py — 最终版

from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from core.memory.retriever import MemoryRetriever
from core.memory.database import MemoryDatabase
from core.client import LLMClient
from core.tools.registry import get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

BASE_SYSTEM = """你是 Mnemis，一个拥有长期记忆的 AI 研究助手。
你了解用户的背景、偏好和目标，回答时主动运用这些信息。
用户问起过去的事情时，主动使用 recall_memories 工具查询记忆。"""


class MemoryEnabledAgent:

    def __init__(self, max_steps: int = 8, tool_names: list[str] | None = None):
        self.llm = LLMClient()
        self.db = MemoryDatabase()
        self.episodic = EpisodicMemory(self.db, self.llm)
        self.semantic = SemanticMemory(self.db, self.llm)
        self.retriever = MemoryRetriever(self.db, self.semantic, self.episodic)

    def start_session(self):
        self.episodic.start_session()

    def chat(self, user_input: str) -> str:
        if not self.episodic._current_session:
            self.episodic.start_session()

        self.episodic.record_message("user", user_input)

        # 用检索器获取最优记忆组合
        retrieval = self.retriever.retrieve_for_context(
            current_query=user_input,
            max_semantic_memories=15,
            max_episodic_summaries=3,
        )
        memory_context = self.retriever.format_memory_context(retrieval)
        system = BASE_SYSTEM
        if memory_context:
            system += f"\n\n{memory_context}"

        context_messages = self.episodic.build_context_messages()
        result = self._run_with_context(context_messages, system)
        self.episodic.record_message("assistant", result)
        return result

    def end_session(self) -> dict:
        session_id = (
            self.episodic._current_session.id
            if self.episodic._current_session
            else None
        )
        summary = self.episodic.end_session()
        new_memories = []
        if session_id:
            messages = self.db.get_session_messages(session_id)
            new_memories = self.semantic.extract_from_conversation(messages, session_id)

        # 每次会话结束时执行时间衰减（轻量级，通常几十毫秒）
        from core.memory.conflict import MemoryEvolution

        evolution = MemoryEvolution(self.db, self.semantic)
        decay_stats = evolution.apply_time_decay()

        return {
            "summary": summary,
            "new_memories": len(new_memories),
            "memory_details": [(m.memory_type.value, m.content) for m in new_memories],
            "decay_stats": decay_stats,
        }

    def _run_with_context(self, context_messages: list[dict], system: str) -> str:
        registry = get_registry()
        tools = registry.get_schemas()

        messages, tool_calls = self.llm.chat_with_tools(
            messages=context_messages,
            tools=tools,
            system=system,
        )

        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                result = registry.execute(tc.name, tc.input)
                result.tool_use_id = tc.id
                tool_results.append(result)
            messages = self.llm.append_tool_results(messages, tool_results)
            messages, _ = self.llm.chat_with_tools(
                messages=messages,
                tools=tools,
                system=system,
            )

        final = ""
        for block in messages[-1].get("content", []):
            if hasattr(block, "text"):
                final += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                final += block.get("text", "")
        return final
