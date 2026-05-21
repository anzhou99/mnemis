# hybrid_test.py
from core.knowledge_base import KnowledgeBase
from core.retriever import HybridSearcher, format_context_with_citations

def main():
    kb = KnowledgeBase()

    # 先确认知识库有内容（复用 kb_test.py 里写入的数据）
    stats = kb.get_stats()
    print(f"知识库状态：{stats}\n")
    if stats["total_points"] == 0:
        print("知识库为空，请先运行 kb_test.py 写入数据")
        return

    searcher = HybridSearcher(kb)

    # ── 对比测试：向量检索 vs Hybrid Search ──────────────────────
    test_cases = [
        {
            "query": "BM25 是什么算法",
            "note": "精确技术术语——向量检索可能找不到，BM25 能精确匹配"
        },
        {
            "query": "如何让 AI 访问外部知识",
            "note": "语义查询——两种方式都能找到，但表述方式不同"
        },
        {
            "query": "lru_cache 的使用方法",
            "note": "代码术语——BM25 能精确匹配函数名"
        },
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"查询：「{case['query']}」")
        print(f"说明：{case['note']}")

        # 纯向量检索
        vector_results = kb.search(case["query"], top_k=3, score_threshold=0.0)
        print(f"\n📐 向量检索 Top-3：")
        for r in vector_results[:3]:
            print(f"  score={r['score']:.3f} | {r['source']} | {r['text'][:60]}...")

        # Hybrid Search
        hybrid_results = searcher.search(case["query"], top_k=3)
        print(f"\n🔀 Hybrid Search Top-3：")
        for r in hybrid_results[:3]:
            tag = {"both": "⚡双命中", "vector": "📐向量", "bm25": "🔤关键词"}[r["match_type"]]
            print(f"  rrf={r['rrf_score']:.5f} | {tag} | {r['source']} | {r['text'][:60]}...")

    # ── 引用溯源演示 ──────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("引用溯源演示\n")

    query = "Python 性能优化有哪些常用手段"
    results = searcher.search(query, top_k=3)
    context_str, citations = format_context_with_citations(results)

    print(f"查询：{query}\n")
    print("检索到的上下文（将注入 prompt）：")
    print("─" * 40)
    print(context_str[:600] + "...")
    print("─" * 40)
    print("\n引用列表：")
    for c in citations:
        print(f"  [{c['index']}] {c['source']} | "
              f"score={c['score']:.5f} | "
              f"来源：{c['match_type']}")
        print(f"       预览：{c['text_preview']}")

if __name__ == "__main__":
    main()