from core.knowledge_base import KnowledgeBase


def insertText(kb: KnowledgeBase):
    # ── 写入测试文档 ──────────────────────────────────────────────
    print("=== 写入单条文本 ===\n")

    #     # 直接写入文本（不需要真实文件）
    doc = {
        "text": """Claude 3.5 Sonnet 在推理能力上比 GPT-4o 有显著提升，尤其是在代码生成任务上。""",
        "source": "claude.md",
    }

    count = kb.add_text(doc["text"], source=doc["source"])
    print(f"  ✓ {doc['source']} → {count} chunks")


def insertTexts(kb: KnowledgeBase):
    # ── 写入测试文档 ──────────────────────────────────────────────
    print("=== 写入文本 ===\n")

    #     # 直接写入文本（不需要真实文件）
    docs = [
        {
            "text": """Python 性能优化的核心是找到瓶颈。使用 cProfile 进行性能分析：
                import cProfile
                cProfile.run('my_function()')
                常见优化手段包括：使用列表推导式替代循环、
                使用 lru_cache 缓存纯函数、使用 numpy 替代纯 Python 数值计算。
                对于 I/O 密集型任务，asyncio 能大幅提升并发性能。""",
            "source": "python_optimization.md",
        },
        {
            "text": """RAG（检索增强生成）是一种让 LLM 能访问外部知识的技术。
                核心思路：在调用 LLM 之前，先从知识库检索相关文档片段，
                把这些片段作为上下文塞进 prompt，让模型基于这些内容生成回答。
                RAG 解决了 LLM 知识截止日期的问题，也降低了幻觉的风险。""",
            "source": "rag_introduction.md",
        },
        {
            "text": """向量数据库专门用于存储和检索高维向量。
                Qdrant 是一个高性能的开源向量数据库，支持 HNSW 索引。
                主要操作：create_collection 创建集合，upsert 写入向量，search 检索最近邻。
                每个向量点（Point）包含：id、vector、payload 三部分。""",
            "source": "qdrant_guide.md",
        },
        {
            "text": """今天上海天气晴朗，气温 22 度，适合户外活动。
                明天预计有小雨，出行请携带雨具。本周气温整体稳定，
                周末会有一次冷空气南下，气温下降约 5 度。""",
            "source": "weather_news.md",
        },
    ]

    for doc in docs:
        count = kb.add_text(doc["text"], source=doc["source"])
        print(f"  ✓ {doc['source']} → {count} chunks")


def insertDocuments(kb: KnowledgeBase):
    # ── 写入测试文档 ──────────────────────────────────────────────
    print("=== 写入文档 ===\n")

    kb.add_document("workspace/AgentHub_架构指南.md")


def delete_source(kb: KnowledgeBase):

    kb.delete_source("python_optimization.md")
    kb.delete_source("qdrant_guide.md")
    kb.delete_source("rag_introduction.md")
    kb.delete_source("weather_news.md")

    stats = kb.get_stats()

    print(stats)


def query(kb: KnowledgeBase):

    # ── 语义检索测试 ──────────────────────────────────────────────
    print("=== 语义检索测试 ===\n")

    queries = [
        # "如何提升 Python 代码性能",  # 应该命中 python_optimization
        # "什么是检索增强生成",  # 应该命中 rag_introduction
        # "向量数据库怎么存数据",  # 应该命中 qdrant_guide
        # "明天需要带伞吗",  # 应该命中 weather_news
        # "机器学习模型训练技巧",  # 知识库里没有，应该返回低分或空
        # "AgentHub 是一个 AI Agent 管理与运行平台。用户可以创建知识库、配置工具、发起对话，并让 Agent 自主规划",  #
        "Claude 3.5 Sonnet 和 GPT-4o 哪个更好",
        "Claude 3.5 Sonnet 的 model string 是什么"
    ]

    for query in queries:
        print(f"查询：「{query}」")
        results = kb.search(query, top_k=2)

        if not results:
            print("  → 无相关结果（相似度低于阈值）\n")
            continue

        for i, r in enumerate(results, 1):
            bar = "█" * int(r["score"] * 20)
            print(f"  [{i}] {r['source']} | 相似度: {r['score']} {bar}")
            print(f"      {r['text'][:80]}...\n")


def query_filter(kb: KnowledgeBase):

    # # ── Filter 检索测试 ───────────────────────────────────────────
    print("=== Filter 检索：只在特定文档里搜索 ===\n")
    results = kb.search(
        "性能", top_k=3, source_filter="python_optimization.md"  # 只在这个文档里搜
    )
    print(f"只在 python_optimization.md 里搜「性能」：{len(results)} 条结果")
    for r in results:
        print(f"  score={r['score']} | {r['text'][:60]}...")


def main():
    kb = KnowledgeBase()

    print(f"\n知识库状态：{kb.get_stats()}")
    print(f"文档来源：{kb.list_sources()}\n")

    insertText(kb)

    # insertTexts(kb)

    # insertDocuments(kb)

    query(kb)

    # query_filter(kb)

    # delete_source(kb)


if __name__ == "__main__":
    main()
