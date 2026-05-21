import jieba
from rank_bm25 import BM25Okapi
from core.knowledge_base import KnowledgeBase
# from core.embeddings import embed_text
from utils.logger import get_logger


logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """
    中英文混合分词。
    中文用 jieba，英文按空格切分，全部转小写。
    """

    # jieba 对中文分词，对英文也能处理
    tokens = list(jieba.cut(text))
    # 过滤掉纯标点和单字符噪音
    tokens = [t.lower().strip() for t in tokens if len(t.strip()) > 1]

    return tokens


class HybridSearcher:
    """
    Hybrid Search 实现：向量检索 + BM25 关键词检索 + RRF 融合。

    注意：BM25 是在内存里对所有 Chunk 建索引的，
    所以需要先把知识库里的文本全部加载进来。
    对于大型知识库（>10万 chunks），需要考虑分片策略。
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """
        从 Qdrant 加载所有 chunks，在内存里构建 BM25 索引。
        当知识库更新时需要重新调用这个方法。
        """
        logger.info("Building BM25 index from knowledge base...")

        # 从 Qdrant scroll 出所有 chunks
        chunks = []
        offset = None
        while True:
            results, offset = self.kb._client.scroll(
                collection_name=self.kb._collection,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                if point.payload:
                    chunks.append(
                        {
                            "id": str(point.id),
                            "text": point.payload["text"],
                            "source": point.payload.get("source", ""),
                            "page_num": point.payload.get("page_num"),
                            "chunk_index": point.payload.get("chunk_index"),
                        }
                    )
            if offset is None:
                break

        if not chunks:
            logger.warning("Knowledge base is empty, BM25 index not built")
            return

        self._bm25_chunks = chunks
        tokenized = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(f"BM25 index built: {len(chunks)} chunks")

    def _vector_search(
        self, query: str, top_k: int, source_filter: str | None
    ) -> list[dict]:
        """向量检索，返回带排名的结果"""
        results = self.kb.search(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
            score_threshold=0.0,  # hybrid search 里不做阈值过滤，交给 RRF
        )
        # 加上排名信息
        for rank, r in enumerate(results, 1):
            r["vector_rank"] = rank
        return results

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        source_filter: str | None,
    ) -> list[dict]:
        """BM25 关键词检索，返回带排名的结果"""
        if self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 按分数排序，取 top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[
            : top_k * 2
        ]  # 多取一些，Filter 后可能数量不够

        results = []
        for rank, (idx, score) in enumerate(indexed, 1):
            chunk = self._bm25_chunks[idx]
            # 应用 source_filter
            if source_filter and chunk["source"] != source_filter:
                continue
            if score < 0.01:  # BM25 分数太低说明完全不相关
                continue
            results.append(
                {
                    **chunk,
                    "score": round(score, 4),
                    "bm25_rank": rank,
                }
            )
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _rrf_fusion(
        vector_results: list[dict], bm25_results: list[dict], top_k: int, k: int = 60
    ) -> list[dict]:
        """
        RRF 融合：只看排名，不看分数绝对值。
        k=60 是经典默认值，来自原始论文。
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        # 向量检索结果贡献分数
        for r in vector_results:
            doc_id = f"{r["source"]}_{r.get("chunk_index",0)}"
            rank = r.get("vector_rank", 999)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            chunk_map[doc_id] = r

        # BM25 结果贡献分数
        for r in bm25_results:
            doc_id = f"{r["source"]}_{r.get("chunk_index",0)}"
            rank = r.get("bm25_rank", 999)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in chunk_map:
                chunk_map[doc_id] = r

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, rrf_score in sorted_ids[:top_k]:
            chunk = chunk_map[doc_id].copy()
            chunk["rrf_score"] = round(rrf_score, 6)
            # 标记这个结果来自哪路检索
            in_vector = any(
                f"{r['source']}_{r.get('chunk_index',0)}" == doc_id
                for r in vector_results
            )
            in_bm25 = any(
                f"{r['source']}_{r.get('chunk_index',0)}" == doc_id
                for r in bm25_results
            )
            chunk["match_type"] = (
                "both" if in_vector and in_bm25 else "vector" if in_vector else "bm25"
            )
            results.append(chunk)

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        score_threshold: float = 0.001,  # RRF 分数阈值
    ) -> list[dict]:
        """
        Hybrid Search 统一入口。
        """
        logger.debug(f"Hybrid search: '{query[:50]}' top_k={top_k}")

        vector_results = self._vector_search(query, top_k * 2, source_filter)
        bm25_results = self._bm25_search(query, top_k * 2, source_filter)

        fused = self._rrf_fusion(vector_results, bm25_results, top_k)

        # 过滤低分结果
        fused = [r for r in fused if r["rrf_score"] >= score_threshold]

        logger.debug(
            f"Hybrid results: {len(fused)} | "
            f"vector={sum(1 for r in fused if r['match_type'] in ('vector','both'))} | "
            f"bm25={sum(1 for r in fused if r['match_type'] in ('bm25','both'))}"
        )
        return fused

    def rebuild_index(self):
        """知识库更新后调用，重建 BM25 索引"""
        self._build_bm25_index()

    # core/retriever.py — 继续追加


def format_context_with_citations(
    results: list[dict],
    max_tokens: int = 3000,
) -> tuple[str, list[dict]]:
    """
    把检索结果格式化成带编号引用的上下文字符串。

    返回：
    - context_str：注入 prompt 的上下文文本
    - citations：引用列表（用于在回答中显示来源）

    示例输出：
    [1] 来源：python_optimization.md（第3页）
    使用 lru_cache 装饰器可以缓存纯函数的计算结果...

    [2] 来源：rag_introduction.md
    RAG 的核心思路是在调用 LLM 之前先检索...
    """
    from core.chunking import count_tokens

    context_parts = []
    citations = []
    total_tokens = 0

    for i, result in enumerate(results, 1):
        # 构造引用标注
        source_info = result["source"]
        if result.get("page_num"):
            source_info += f"（第 {result['page_num']} 页）"

        citation_header = f"[{i}] 来源：{source_info}"
        chunk_text = result["text"]
        entry = f"{citation_header}\n{chunk_text}"
        entry_tokens = count_tokens(entry)

        # 控制总长度，避免撑爆 context window
        if total_tokens + entry_tokens > max_tokens:
            logger.debug(f"Context truncated at {i-1} chunks ({total_tokens} tokens)")
            break

        context_parts.append(entry)
        citations.append(
            {
                "index": i,
                "source": result["source"],
                "page_num": result.get("page_num"),
                "score": result.get("rrf_score", result.get("score", 0)),
                "match_type": result.get("match_type", "vector"),
                "text_preview": chunk_text[:100] + "...",
            }
        )
        total_tokens += entry_tokens

    context_str = "\n\n".join(context_parts)
    return context_str, citations


def build_rag_prompt(
    query: str,
    context_str: str,
) -> str:
    """
    构造注入了检索结果的 RAG prompt。
    """
    return f"""请基于以下参考资料回答问题。

## 参考资料
{context_str}

## 问题
{query}

## 要求
- 只使用参考资料中的信息作答
- 在回答中用 [数字] 标注信息来源，如「根据文档 [1]...」
- 如果参考资料不足以回答问题，明确说明
- 不要编造参考资料中没有的内容"""
