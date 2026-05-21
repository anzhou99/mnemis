# core/memory/conflict.py
import uuid
import math
from datetime import datetime, timedelta
from typing import Literal
from core.memory.database import MemoryDatabase
from core.memory.models import Memory, MemoryType, MemoryConflict
from core.memory.semantic import SemanticMemory, MEMORY_COLLECTION
from core.client import LLMClient
from core.models import Message as LLMMessage
from core.embeddings import Embeddings, cosine_similarity
from utils.logger import get_logger

logger = get_logger(__name__)

# 触发冲突检测的相似度阈值
CONFLICT_SIMILARITY_THRESHOLD = 0.85

# 触发遗忘的置信度阈值
FORGET_THRESHOLD = 0.15

# 各类型记忆的衰减半衰期（天）
HALF_LIFE_DAYS = {
    MemoryType.PREFERENCE: 90,
    MemoryType.FACT: 180,
    MemoryType.BACKGROUND: 60,
    MemoryType.GOAL: 45,
}

CONFLICT_JUDGE_SYSTEM = """你是记忆冲突判断专家。
分析两条记忆之间的关系，给出判断。

判断类型：
- contradiction：直接矛盾（一条说A，另一条说非A）
- update：版本更新（新记忆是旧记忆的进化，不矛盾但旧的已过时）
- overlap：语义重叠（两条说的是同一件事，可以合并）
- no_conflict：没有实质冲突，可以共存

只输出 JSON，不要任何其他文字：
{"conflict_type": "contradiction|update|overlap|no_conflict", "reason": "一句话说明", "merged_content": "如果是overlap，给出合并后的内容；否则为null"}"""


class ConflictResolver:
    """
    记忆冲突检测与解决器。
    在新记忆写入前调用，检查是否与现有记忆冲突。
    """

    def __init__(
        self,
        db: MemoryDatabase,
        semantic: SemanticMemory,
        llm: LLMClient,
    ):
        self.db = db
        self.semantic = semantic
        self.llm = llm

    # ── 核心：新记忆写入前的冲突检测 ─────────────────────────────

    def check_and_resolve(
        self,
        new_memory: Memory,
        new_vector: list[float],
    ) -> tuple[bool, Memory]:
        """
        检查新记忆是否与现有记忆冲突，并执行解决策略。

        返回：
        - (True, memory)：记忆应该被写入（可能是修改后的版本）
        - (False, memory)：记忆不应该被写入（被现有记忆覆盖或合并了）
        """
        # 在 Qdrant 里找相似度 > 阈值的现有记忆
        similar = self._find_similar_memories(new_vector, new_memory.memory_type)

        if not similar:
            # 没有相似记忆，直接写入
            return True, new_memory

        # 和最相似的记忆进行冲突判断
        most_similar, sim_score = similar[0]
        logger.debug(
            f"Conflict check: new='{new_memory.content[:30]}' "
            f"vs existing='{most_similar.content[:30]}' "
            f"(similarity={sim_score:.3f})"
        )

        conflict_result = self._judge_conflict(new_memory, most_similar)
        conflict_type = conflict_result.get("conflict_type", "no_conflict")

        if conflict_type == "no_conflict":
            return True, new_memory

        elif conflict_type == "contradiction":
            return self._resolve_contradiction(new_memory, most_similar)

        elif conflict_type == "update":
            return self._resolve_update(new_memory, most_similar)

        elif conflict_type == "overlap":
            return self._resolve_overlap(
                new_memory, most_similar, conflict_result.get("merged_content")
            )

        return True, new_memory

    def _find_similar_memories(
        self,
        vector: list[float],
        memory_type: MemoryType,
        top_k: int = 3,
    ) -> list[tuple[Memory, float]]:
        """在 Qdrant 里找相似度超过阈值的现有记忆"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self.semantic._qdrant.query_points(
            collection_name=MEMORY_COLLECTION,
            query=vector,
            limit=top_k,
            score_threshold=CONFLICT_SIMILARITY_THRESHOLD,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="memory_type", match=MatchValue(value=memory_type.value)
                    )
                ]
            ),
            with_payload=True,
        )

        similar = []
        for hit in results.points:
            memory_id = hit.payload.get("memory_id") if hit.payload else ''
            if not memory_id:
                continue
            existing = self.semantic._get_memory_by_id(memory_id)
            if existing:
                similar.append((existing, hit.score))

        return similar

    def _judge_conflict(
        self,
        new_memory: Memory,
        existing_memory: Memory,
    ) -> dict:
        """用 LLM 判断两条记忆的关系"""
        prompt = f"""分析以下两条记忆的关系：

新记忆：「{new_memory.content}」
现有记忆：「{existing_memory.content}」（创建于 {existing_memory.created_at.strftime('%Y-%m-%d')}）

判断它们的关系类型并给出 JSON 结果。"""

        try:
            response = self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                system=CONFLICT_JUDGE_SYSTEM,
                temperature=0.1,
            )
            import json

            raw = response.content.strip()
            raw = (
                raw.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Conflict judgment failed: {e}")
            return {"conflict_type": "no_conflict", "reason": "判断失败，保留双方"}

    def _resolve_contradiction(
        self,
        new_memory: Memory,
        old_memory: Memory,
    ) -> tuple[bool, Memory]:
        """矛盾解决：新覆盖旧（用户改变了想法）"""
        logger.info(
            f"Contradiction resolved: new wins | "
            f"new='{new_memory.content[:40]}' | "
            f"old='{old_memory.content[:40]}'"
        )

        # 旧记忆置信度降为 0（软删除，保留历史）
        old_memory.confidence = 0.0
        self.db.update_memory(old_memory)

        # 记录冲突日志
        self._log_conflict(new_memory.id, old_memory.id, "contradiction", "new_wins")

        return True, new_memory  # 新记忆正常写入

    def _resolve_update(
        self,
        new_memory: Memory,
        old_memory: Memory,
    ) -> tuple[bool, Memory]:
        """版本更新：两者保留，旧记忆置信度降低"""
        logger.info(f"Update resolved: both kept | " f"new='{new_memory.content[:40]}'")

        # 旧记忆置信度小幅降低，但不清零（仍有参考价值）
        old_memory.confidence = max(0.3, old_memory.confidence * 0.7)
        self.db.update_memory(old_memory)

        self._log_conflict(new_memory.id, old_memory.id, "update", "both_kept")

        return True, new_memory  # 新记忆正常写入

    def _resolve_overlap(
        self,
        new_memory: Memory,
        old_memory: Memory,
        merged_content: str | None,
    ) -> tuple[bool, Memory]:
        """语义重叠：合并为一条，淘汰旧条"""
        if merged_content:
            # 用合并后的内容更新旧记忆
            old_memory.content = merged_content
            old_memory.confidence = max(new_memory.confidence, old_memory.confidence)
            self.db.update_memory(old_memory)

            logger.info(f"Overlap merged: '{merged_content[:50]}'")
            self._log_conflict(new_memory.id, old_memory.id, "overlap", "old_wins")

            # 新记忆不再单独写入（已合并进旧记忆）
            return False, old_memory
        else:
            # 没有合并内容，按 update 处理
            return self._resolve_update(new_memory, old_memory)

    def _log_conflict(
        self,
        new_id: str,
        old_id: str,
        conflict_type: Literal['contradiction', 'update', 'overlap'],
        resolution: Literal['new_wins', 'old_wins', 'both_kept'],
    ):
        """记录冲突处理日志"""
        conflict = MemoryConflict(
            id=str(uuid.uuid4()),
            new_memory_id=new_id,
            old_memory_id=old_id,
            conflict_type=conflict_type,
            resolution=resolution,
        )
        with self.db._conn() as conn:
            conn.execute(
                """INSERT INTO memory_conflicts
                   VALUES (?,?,?,?,?,?)""",
                (
                    conflict.id,
                    conflict.new_memory_id,
                    conflict.old_memory_id,
                    conflict.conflict_type,
                    conflict.resolution,
                    conflict.resolved_at.isoformat(),
                ),
            )


class MemoryEvolution:
    """
    记忆演进管理器。
    负责：置信度时间衰减 / 记忆遗忘 / 定期维护
    """

    def __init__(self, db: MemoryDatabase, semantic: SemanticMemory):
        self.db = db
        self.semantic = semantic

    def apply_time_decay(self) -> dict:
        """
        对所有记忆应用时间衰减。
        建议每天调用一次（或每次会话结束时调用）。
        返回衰减统计。
        """
        all_memories = self.db.get_memories_by_type(
            min_confidence=0.01,  # 取几乎所有记忆
            limit=1000,
        )

        decayed = 0
        forgotten = 0
        now = datetime.now()

        for memory in all_memories:
            half_life = HALF_LIFE_DAYS.get(memory.memory_type, 90)
            days_elapsed = (now - memory.created_at).days

            if days_elapsed <= 0:
                continue

            # 指数衰减：confidence × (0.5 ^ (days / half_life))
            decay_factor = 0.5 ** (days_elapsed / half_life)
            new_confidence = memory.confidence * decay_factor

            if new_confidence < FORGET_THRESHOLD:
                # 置信度太低，执行遗忘
                self._forget_memory(memory)
                forgotten += 1
            elif abs(new_confidence - memory.confidence) > 0.01:
                # 有显著变化，更新
                memory.confidence = round(new_confidence, 4)
                self.db.update_memory(memory)
                decayed += 1

        logger.info(f"Time decay applied: {decayed} decayed, {forgotten} forgotten")
        return {"decayed": decayed, "forgotten": forgotten, "total": len(all_memories)}

    def _forget_memory(self, memory: Memory):
        """
        遗忘一条记忆：从 SQLite 和 Qdrant 双删除。
        """
        logger.debug(
            f"Forgetting memory: [{memory.memory_type.value}] {memory.content[:40]}"
        )

        # 从 Qdrant 删除向量
        if memory.vector_id:
            try:
                self.semantic._qdrant.delete(
                    collection_name=MEMORY_COLLECTION,
                    points_selector=[memory.vector_id],
                )
            except Exception as e:
                logger.warning(f"Failed to delete vector {memory.vector_id}: {e}")

        # 从 SQLite 删除记录
        self.db.delete_memory(memory.id)

    def forget_stale_memories(
        self,
        days_threshold: int = 90,
        min_access_count: int = 3,
    ) -> int:
        """
        遗忘长期未访问的低价值记忆。
        条件：超过 days_threshold 天未访问 AND 访问次数 < min_access_count
        """
        all_memories = self.db.get_memories_by_type(limit=1000)
        forgotten = 0

        for memory in all_memories:
            if (
                memory.is_stale(days_threshold)
                and memory.access_count < min_access_count
            ):
                self._forget_memory(memory)
                forgotten += 1

        logger.info(f"Stale memory cleanup: {forgotten} forgotten")
        return forgotten

    def consolidate_memories(self, llm: LLMClient) -> int:
        """
        记忆整合：把多条碎片化记忆合并成更完整的描述。
        适合定期（如每周）运行，提升记忆质量。
        返回合并的记忆对数。
        """
        consolidated = 0

        for mem_type in MemoryType:
            memories = self.db.get_memories_by_type(
                memory_type=mem_type,
                min_confidence=0.5,
                limit=20,
            )
            if len(memories) < 3:
                continue

            # 让 LLM 判断哪些记忆可以合并
            contents = [m.content for m in memories]
            result = self._ask_llm_to_consolidate(contents, mem_type.value, llm)

            for merge_group in result.get("merge_groups", []):
                if len(merge_group["indices"]) < 2:
                    continue

                # 找到要合并的记忆
                to_merge = [
                    memories[i] for i in merge_group["indices"] if i < len(memories)
                ]
                if len(to_merge) < 2:
                    continue

                # 保留置信度最高的，删除其余的，更新内容
                to_merge.sort(key=lambda m: m.confidence, reverse=True)
                keeper = to_merge[0]
                keeper.content = merge_group["merged_content"]
                keeper.confidence = max(m.confidence for m in to_merge)
                self.db.update_memory(keeper)

                for obsolete in to_merge[1:]:
                    self._forget_memory(obsolete)
                    consolidated += 1

        logger.info(f"Memory consolidation: {consolidated} memories merged")
        return consolidated

    def _ask_llm_to_consolidate(
        self,
        contents: list[str],
        mem_type: str,
        llm: LLMClient,
    ) -> dict:
        """让 LLM 识别可以合并的记忆组"""
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(contents))
        prompt = f"""以下是关于用户的 {mem_type} 类记忆：

{numbered}

找出内容高度相似、可以合并的记忆组。
只输出 JSON：
{{
  "merge_groups": [
    {{"indices": [0, 2], "merged_content": "合并后的统一描述"}},
    ...
  ]
}}
如果没有可合并的，返回 {{"merge_groups": []}}"""

        try:
            response = llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                system="你是记忆整合专家，帮助识别和合并重复的用户记忆。只输出 JSON。",
                temperature=0.1,
            )
            import json

            raw = response.content.strip()
            raw = (
                raw.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Consolidation LLM call failed: {e}")
            return {"merge_groups": []}
