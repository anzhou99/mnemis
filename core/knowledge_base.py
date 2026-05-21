import asyncio
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
from core.config import settings
from core.chunking import Chunk, chunk_document
from core.embeddings import Embeddings
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBase:
    """
    向量知识库的统一接口
    封装Qrdant 的所有操作：创建集合、写入文档、语义检索
    """

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self._collection = settings.qdrant_collection
        self._ensure_collection()
        self._embedding_client = Embeddings()

    # ————初始化————————————————————————————————————————
    def _ensure_collection(self):
        """如果 Collection 不存在则创建，已存在则跳过"""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            logger.debug(f"Collection '{self._collection}' already exists")
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )
        logger.info(f"Created collection '{self._collection}'")

    # ———写入—————————+————————————————————————————————————

    def add_document(self, file_path: str, **kwargs) -> int:
        """
        添加一个文档到知识库。
        完整流程：读文件 → Chunking → Embedding → 存入 Qdrant
        返回写入的 Chunk 数量。
        """
        logger.info(f"Adding document: {file_path}")

        # 切割文档
        chunks = chunk_document(file_path=file_path, **kwargs)
        if not chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return 0

        return self._add_chunks(chunks)

    def add_text(self, text: str, source: str = "manual") -> int:
        """
        直接添加文本（不经过文件读取），适合运行时动态添加内容。
        """
        from core.chunking import RecursiveChunker

        chunker = RecursiveChunker()
        chunks = chunker.chunk_text(text, source=source)

        return self._add_chunks(chunks)

    def _add_chunks(self, chunks: list[Chunk]) -> int:
        """批量向量化并写入 Qdrant"""

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        logger.debug(f"Embedding {len(texts)} chunks...")
        vectors = self._embedding_client.embed_batch(texts)

        # 构造 Qdrant Point 列表
        points = []
        for chunk, vector in zip(chunks, vectors):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "page_num": chunk.page_num,
                    "section": chunk.section,
                },
            )
            points.append(point)

        # 分批写入（Qdrant 推荐每批不超过 100 个）
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)

            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} points")

        logger.info(f"Added {len(chunks)} chunks from '{chunks[0].source}'")
        return len(chunks)

    # ── 检索 ──────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,  # 只在特定文档里搜索
        score_threshold: float = 0.3,  # 过滤掉相似度太低的结果
    ) -> list[dict]:
        """
        语义检索。
        返回 top_k 个最相关的 Chunk，按相似度降序排列。
        """
        # 查询也要向量化
        query_vector = self._embedding_client.embed_text(query)

        # 构造过滤条件（可选）
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source_filter))
                ]
            )
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,  # 返回 payload（原文和元数据）
        )

        # 格式化返回结果
        formatted = []
        for score, payload in [(p.score, p.payload) for p in results.points]:
            if payload:
                formatted.append(
                    {
                        "text": payload["text"],
                        "source": payload["source"],
                        "page_num": payload["page_num"],
                        "chunk_index": payload["chunk_index"],
                        "score": round(score, 4),
                    }
                )

        logger.debug(
            f"Search '{query[:40]}...' → {len(formatted)} results "
            f"(top score: {formatted[0]['score'] if formatted else 'N/A'})"
        )
        return formatted

    # ── 管理 ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        info = self._client.get_collection(self._collection)

        return {
            "total_points": info.points_count,
            "collection": self._collection,
            "vector_size": info.config.params.vectors.size,
        }

    def list_sources(self) -> list[str]:
        """列出知识库中所有文档来源"""
        # 用 scroll 遍历所有 payload，提取唯一的 source 值
        sources = set()
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=self._collection,
                limit=100,
                offset=offset,
                with_payload=["source"],
            )
            for point in results:
                sources.add(point.payload.get("source", "unknown"))
            if offset is None:
                break
        return sorted(sources)

    def delete_source(self, source: str) -> int:
        """删除来自特定文档的所有 Chunks"""
        result = self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        logger.info(f"Deleted all chunks from source: {source}")
        return result.status
