import asyncio
import re
from core.embeddings import Embeddings, cosine_similarity
from core.chunking import RecursiveChunker, chunk_document


def explore():
    embedding_client = Embeddings()
    print("=== 实验1：语义近似的文本，向量相似度高 ===")

    texts = [
        "你说了什么",
        "你们讨论的啥",
        "你谈的啥",
        "你吃了什么",
        "你走了没",
    ]

    print("=== 向量优化中... ===")

    vectors = {text: embedding_client.embed_text(text) for text in texts}

    query = texts[0]  # "Python 性能优化技巧"
    print(f"\n查询：「{query}」\n")
    print(f"{'文本':<30} {'相似度':>8}")
    print("─" * 42)

    results = []

    for text, vec in vectors.items():
        if text == query:
            continue
        sim = cosine_similarity(vectors[query], vec)
        results.append((text, sim))

    # 按相似度排序
    for text, sim in sorted(results, key=lambda x: x[1], reverse=True):
        bar = "█" * int(sim * 20)
        print(f"{text:<30} {sim:.4f}  {bar}")

    print("\n=== 实验2：同一个词的不同语义 ===\n")
    # 展示 Embedding 能区分同词不同义
    context_pairs = [
        ("苹果公司发布了新款手机", "iPhone 是苹果的旗舰产品"),  # 科技含义
        ("苹果公司发布了新款手机", "苹果富含维生素 C"),  # 水果含义
    ]
    for text_a, text_b in context_pairs:
        vec_a = embedding_client.embed_text(text_a)
        vec_b = embedding_client.embed_text(text_b)
        sim = cosine_similarity(vec_a, vec_b)
        print(f"「{text_a}」")
        print(f"「{text_b}」")
        print(f"→ 相似度：{sim:.4f}\n")

    print("=== 实验3：向量的维度和数值 ===\n")
    vec = embedding_client.embed_text("Hello world")
    print(f"维度：{len(vec)}")
    print(f"前10个值：{[round(v, 4) for v in vec[:10]]}")
    print(f"值域：[{min(vec):.4f}, {max(vec):.4f}]")


def embedding_document():

    embedding_client = Embeddings()
    chunks = chunk_document("workspace/AgentHub_架构指南.md")

    print(len(chunks))

    all_vectors = embedding_client.embed_batch(items_list=[c.text for c in chunks])

    print(len(all_vectors))


if __name__ == "__main__":
    # explore()
    embedding_document()
