# core/memory/retriever.py
from core.memory.database import MemoryDatabase
from core.memory.semantic import SemanticMemory
from core.memory.episodic import EpisodicMemory
from core.memory.models import Memory, MemoryType
from core.embeddings import Embeddings, cosine_similarity
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryRetriever:
    """
    记忆检索器：综合情景记忆和语义记忆，
    为每次对话构建最优的记忆上下文。
    """

    def __init__(
        self,
        db: MemoryDatabase,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
    ):
        self.db = db
        self.semantic = semantic
        self.episodic = episodic

    def retrieve_for_context(
        self,
        current_query: str,
        max_semantic_memories: int = 15,
        max_episodic_summaries: int = 3,
        min_confidence: float = 0.4,
    ) -> dict:
        """
        为当前查询检索最相关的记忆组合。

        返回字典包含：
        - semantic_memories: 语义记忆列表
        - episodic_summaries: 相关历史会话摘要列表
        - total_token_estimate: 预估 token 消耗
        """
        # 1. 语义记忆：全量取（数量少，全部有价值）
        semantic_memories = self.semantic.get_all_by_type(
            min_confidence=min_confidence,
            limit=max_semantic_memories,
        )

        # 2. 情景记忆：按当前话题相关性筛选
        episodic_summaries = self._retrieve_relevant_episodes(
            query=current_query,
            top_k=max_episodic_summaries,
        )

        # 估算 token 消耗（粗略：中文约 1.5 字符/token）
        semantic_chars = sum(len(m.content) for m in semantic_memories)
        episodic_chars = sum(len(s) for s in episodic_summaries)
        total_estimate = int((semantic_chars + episodic_chars) / 1.5)

        logger.debug(
            f"Memory retrieval | semantic={len(semantic_memories)} | "
            f"episodic={len(episodic_summaries)} | "
            f"tokens≈{total_estimate}"
        )

        return {
            "semantic_memories": semantic_memories,
            "episodic_summaries": episodic_summaries,
            "total_token_estimate": total_estimate,
        }

    def _retrieve_relevant_episodes(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        从历史会话摘要中，找和当前查询最相关的 Top-K 条。
        用向量相似度来判断相关性。
        """
        sessions = self.db.get_recent_sessions(limit=20)
        sessions_with_summary = [s for s in sessions if s.summary]

        if not sessions_with_summary:
            return []

        if len(sessions_with_summary) <= top_k:
            # 历史会话少，全部返回
            return [s.summary for s in sessions_with_summary if s and s.summary]

        # 向量化查询

        query_vec = Embeddings().embed_text(query)

        # 计算每个摘要和查询的相似度
        scored = []
        summaries = [s.summary for s in sessions_with_summary]

        # 批量向量化摘要（避免逐条调用 API）

        summary_vecs = Embeddings().embed_batch([s for s in summaries if s])

        for session, summary, vec in zip(
            sessions_with_summary, summaries, summary_vecs
        ):
            sim = cosine_similarity(query_vec, vec)
            date_str = session.started_at.strftime("%Y-%m-%d")
            title = session.title or "未命名会话"
            formatted = f"[{date_str}] {title}\n{summary}"
            scored.append((sim, formatted))

        # 按相似度排序，取 Top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def format_memory_context(
        self,
        retrieval_result: dict,
        max_tokens: int = 600,
    ) -> str:
        """
        把检索结果格式化成注入 system prompt 的字符串。
        控制总长度在 max_tokens 以内。
        """
        parts = []
        used_chars = 0
        budget_chars = int(max_tokens * 1.5)  # 粗略字符预算

        # 语义记忆部分
        semantic_str = self.semantic.format_for_injection(
            retrieval_result["semantic_memories"],
            max_tokens=max_tokens // 2,
        )
        if semantic_str:
            parts.append(semantic_str)
            used_chars += len(semantic_str)

        # 情景摘要部分
        for summary in retrieval_result["episodic_summaries"]:
            if used_chars + len(summary) > budget_chars:
                break
            parts.append(summary)
            used_chars += len(summary)

        if not parts:
            return ""

        header = "## 关于用户的记忆\n"
        return header + "\n\n".join(parts)
