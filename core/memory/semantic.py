# core/memory/semantic.py
import json
import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    Range,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from core.memory.database import MemoryDatabase
from core.memory.models import Memory, MemoryType, Message
from core.client import LLMClient
from core.models import Message as LLMMessage
from core.embeddings import Embeddings
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# 语义记忆的 Qdrant collection 名（独立于 P2 的知识库）
MEMORY_COLLECTION = "mnemis_memory"

EXTRACTION_SYSTEM = """你是用户记忆提炼专家。
从对话中识别并提炼关于用户的持久性事实，忽略临时性、无意义的内容。

记忆类型：
- preference：用户的偏好和习惯（如「喜欢简洁回答」「偏好代码示例」）
- fact：关于用户的客观事实（如「是 Python 开发者」「在上海工作」）
- background：用户的背景和经历（如「正在做 AI Agent 项目」「有5年后端经验」）
- goal：用户明确表达的目标（如「想开源项目」「计划写技术博客」）

输出规则：
- 只提炼有长期价值的信息，忽略一次性问题和闲聊
- 每条记忆必须是完整的陈述句
- 置信度：明确表达=1.0，隐含推断=0.7，模糊猜测=0.5
- 如果对话中没有值得提炼的用户信息，返回空列表 []

只输出 JSON 数组，不要任何其他文字。"""


class SemanticMemory:
    """
    语义记忆管理器。
    负责：从对话提炼事实 / 向量化存储 / 按类型和语义检索
    """

    def __init__(self, db: MemoryDatabase, llm_client: LLMClient):
        self.db = db
        self.llm = llm_client
        self.embeddings = Embeddings()
        self._qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._ensure_memory_collection()

        # 延迟初始化 ConflictResolver（避免循环导入）
        self._resolver = None

    def _get_resolver(self):
        if self._resolver is None:
            from core.memory.conflict import ConflictResolver

            self._resolver = ConflictResolver(self.db, self, self.llm)
        return self._resolver

    # ── 初始化 ────────────────────────────────────────────────────

    def _ensure_memory_collection(self):
        """确保 Qdrant 里有专门存语义记忆的 collection"""
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if MEMORY_COLLECTION not in existing:
            self._qdrant.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created memory collection: {MEMORY_COLLECTION}")

    # ── 核心：从对话提炼记忆 ──────────────────────────────────────

    def extract_from_conversation(
        self,
        messages: list[Message],
        session_id: str,
    ) -> list[Memory]:
        """
        从一批对话消息里提炼语义记忆。
        通常在会话结束后调用。
        返回提炼并保存成功的记忆列表。
        """
        if not messages:
            return []

        # 构造对话文本（只用 user 消息，assistant 消息一般不含用户信息）
        dialogue = "\n".join(f"用户: {m.content}" for m in messages if m.role == "user")

        if not dialogue.strip():
            return []

        logger.debug(f"Extracting memories from {len(messages)} messages...")

        # 调用 LLM 提炼
        raw_memories = self._call_extraction_llm(dialogue, session_id)
        if not raw_memories:
            logger.debug("No memories extracted from this conversation")
            return []

        # 保存并向量化
        saved = self._save_memories(raw_memories, session_id)
        logger.info(
            f"Extracted and saved {len(saved)} memories from session {session_id}"
        )
        return saved

    def _call_extraction_llm(
        self,
        dialogue: str,
        session_id: str,
    ) -> list[dict]:
        """调用 LLM 提炼，返回原始字典列表"""
        prompt = f"""从以下用户对话中提炼关于用户的持久性事实。

对话内容：
{dialogue}

请输出 JSON 数组格式的记忆列表。"""

        try:
            response = self.llm.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                system=EXTRACTION_SYSTEM,
                temperature=0.1,  # 提炼任务用低温度，保证格式稳定
            )

            raw = response.content.strip()
            # 清理可能的 markdown 包裹
            raw = (
                raw.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            result = json.loads(raw)

            if not isinstance(result, list):
                logger.warning(f"LLM returned non-list: {type(result)}")
                return []

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse memory extraction result: {e}")
            return []
        except Exception as e:
            logger.error(f"Memory extraction LLM call failed: {e}")
            return []

    def _save_memories(
        self,
        raw_memories: list[dict],
        session_id: str,
    ) -> list[Memory]:
        """升级版：写入前检测冲突"""
        saved = []
        contents = [m.get("content", "") for m in raw_memories if m.get("content")]
        if not contents:
            return []

        vectors = self.embeddings.embed_batch(contents)
        resolver = self._get_resolver()

        for raw, vector in zip(raw_memories, vectors):
            content = raw.get("content", "").strip()
            if not content:
                continue

            try:
                mem_type = MemoryType(raw.get("memory_type", "fact"))
            except ValueError:
                mem_type = MemoryType.FACT

            confidence = float(max(0.0, min(1.0, raw.get("confidence", 1.0))))
            memory_id = str(uuid.uuid4())
            vector_id = str(uuid.uuid4())

            # 构造候选记忆对象
            candidate = Memory(
                id=memory_id,
                content=content,
                memory_type=mem_type,
                confidence=confidence,
                source_session_id=session_id,
                vector_id=vector_id,
            )

            # ← 冲突检测（新增）
            should_save, final_memory = resolver.check_and_resolve(candidate, vector)

            if not should_save:
                logger.debug(f"Memory not saved (conflict resolved): {content[:40]}")
                continue

            # 写入 Qdrant
            self._qdrant.upsert(
                collection_name=MEMORY_COLLECTION,
                points=[
                    PointStruct(
                        id=final_memory.vector_id if final_memory.vector_id else '',
                        vector=vector,
                        payload={
                            "memory_id": final_memory.id,
                            "content": final_memory.content,
                            "memory_type": final_memory.memory_type.value,
                            "confidence": final_memory.confidence,
                            "session_id": session_id,
                        },
                    )
                ],
            )

            # 写入 SQLite
            self.db.save_memory(final_memory)
            saved.append(final_memory)

        return saved

    # ── 检索 ──────────────────────────────────────────────────────

    def search_by_semantic(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.3,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        """
        语义检索：找和查询最相关的记忆。
        适合：「找和当前话题相关的用户信息」
        """
        query_vector = self.embeddings.embed_text(query)

        # 构造 Qdrant Filter
        range = Range(gte=min_confidence)
        filters: list[Condition] = [FieldCondition(key="confidence", range=range)]
        if memory_type:
            filters.append(
                FieldCondition(
                    key="memory_type", match=MatchValue(value=memory_type.value)
                )
            )

        results = self._qdrant.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            limit=top_k,
            query_filter=Filter(must=filters) if filters else None,
            with_payload=True,
            score_threshold=0.4,
        )

        memories = []
        for hit in results.points:
            memory_id = hit.payload.get("memory_id") if hit.payload else ""
            if not memory_id:
                continue
            # 从 SQLite 取完整记忆（含 access_count 等字段）
            mem = self._get_memory_by_id(memory_id)
            if mem:
                self._record_access(mem)
                memories.append(mem)

        return memories

    def get_all_by_type(
        self,
        memory_type: MemoryType | None = None,
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> list[Memory]:
        """
        按类型精确查询：适合「获取所有用户偏好」这类需求。
        """
        memories = self.db.get_memories_by_type(
            memory_type=memory_type,
            min_confidence=min_confidence,
            limit=limit,
        )
        for mem in memories:
            self._record_access(mem)
        return memories

    def format_for_injection(
        self,
        memories: list[Memory],
        max_tokens: int = 400,
    ) -> str:
        """
        把记忆列表格式化成注入 system prompt 的字符串。
        按类型分组，控制总长度。
        """
        if not memories:
            return ""

        # 按类型分组
        grouped: dict[str, list[str]] = {}
        for mem in memories:
            key = mem.memory_type.value
            grouped.setdefault(key, []).append(mem.content)

        # 类型标签的中文映射
        type_labels = {
            "preference": "偏好",
            "fact": "基本信息",
            "background": "背景",
            "goal": "目标",
        }

        parts = []
        for mem_type, contents in grouped.items():
            label = type_labels.get(mem_type, mem_type)
            items = "\n".join(f"  • {c}" for c in contents)
            parts.append(f"【用户{label}】\n{items}")

        result = "\n\n".join(parts)

        # 简单 token 估算（中文约 1.5 字符/token）
        if len(result) > max_tokens * 1.5:
            result = result[: int(max_tokens * 1.5)] + "..."

        return result

    # ── 私有辅助 ──────────────────────────────────────────────────

    def _get_memory_by_id(self, memory_id: str) -> Memory | None:
        """从 SQLite 按 ID 取单条记忆"""
        import sqlite3

        with self.db._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        if row:
            return self.db._row_to_memory(row)
        return None

    def _record_access(self, memory: Memory):
        """记录记忆被访问，更新 last_accessed 和 access_count"""
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        self.db.update_memory(memory)

    def get_stats(self) -> dict:
        """获取语义记忆统计"""
        all_memories = self.db.get_memories_by_type(limit=1000)
        by_type: dict[str, int] = {}
        for m in all_memories:
            by_type[m.memory_type.value] = by_type.get(m.memory_type.value, 0) + 1
        return {
            "total": len(all_memories),
            "by_type": by_type,
            "avg_confidence": round(
                sum(m.confidence for m in all_memories) / max(len(all_memories), 1), 3
            ),
        }
