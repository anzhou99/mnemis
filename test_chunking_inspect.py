# chunking_inspect.py
from pathlib import Path
from core.chunking import chunk_document, RecursiveChunker, count_tokens


def inspect_chunks(chunks, show_n: int = 5):
    """打印 Chunk 的详细信息，方便肉眼检查质量"""
    total_tokens = sum(c.token_count for c in chunks)

    print(f"\n📊 切割统计：")
    print(f"   总 Chunk 数：{len(chunks)}")
    print(f"   总 Token 数：{total_tokens}")
    print(f"   平均 Tokens：{total_tokens // len(chunks)}")
    print(f"   最大 Chunk：{max(c.token_count for c in chunks)} tokens")
    print(f"   最小 Chunk：{min(c.token_count for c in chunks)} tokens")

    print(f"\n📝 前 {show_n} 个 Chunk 详情：")
    for chunk in chunks[:show_n]:
        print(f"\n  ── Chunk {chunk.chunk_index} ──")
        print(
            f"  来源：{chunk.source} | 页码：{chunk.page_num} | Tokens：{chunk.token_count}"
        )
        print(f"  内容：")
        # 每行缩进显示，方便阅读
        for line in chunk.text[:300].split("\n"):
            print(f"    {line}")
        if len(chunk.text) > 300:
            print(f"    ...（共 {len(chunk.text)} 字符）")

    # 检查边界：查看相邻 Chunk 之间的连接是否自然
    if len(chunks) >= 2:
        print(f"\n🔍 边界检查（Chunk 0 末尾 & Chunk 1 开头）：")
        print(f"  Chunk 0 末尾：...{chunks[0].text[-100:]!r}")
        print(f"  Chunk 1 开头：{chunks[1].text[:100]!r}...")


def demo_with_text():
    """用一段内置文本演示，不需要真实文件"""
    sample_text = """
# Python 性能优化指南

## 第一节：理解性能瓶颈

在优化 Python 代码之前，最重要的步骤是找到瓶颈所在。
过早优化是万恶之源。使用 cProfile 或 line_profiler 来定位热点代码。

常见的性能陷阱包括：在循环中重复计算不变的值；使用列表替代集合做成员检测；
不必要的函数调用开销。

## 第二节：内置工具的妙用

Python 的内置函数通常比手写循环快 5-10 倍，因为它们是用 C 实现的。
map()、filter()、zip() 在处理大型序列时效果显著。

列表推导式比等效的 for 循环快约 35%，原因是字节码层面的优化。
但不要为了推导式而推导式，可读性同样重要。

## 第三节：缓存策略

functools.lru_cache 是最简单的缓存手段。对于纯函数（相同输入永远相同输出），
加上 @lru_cache(maxsize=128) 装饰器即可。

更复杂的场景可以考虑 Redis 或 Memcached 作为分布式缓存。
缓存失效是计算机科学中最难的两件事之一（另一件是命名）。
    """

    chunker = RecursiveChunker(max_tokens=150, overlap_tokens=30)
    chunks = chunker.chunk_text(sample_text, source="sample.md")
    inspect_chunks(chunks, show_n=4)


if __name__ == "__main__":
    # demo_with_text()

    # 如果有真实文件，取消注释：
    chunks = chunk_document("workspace/qdrant.pdf")
    inspect_chunks(chunks)
