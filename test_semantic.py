# semantic_demo.py
import core.tools
from core.memory.manager import MemoryEnabledAgent
from core.memory.database import MemoryDatabase
from core.memory.semantic import SemanticMemory
from core.memory.models import MemoryType
from core.client import LLMClient


def session_1_introduce():
    """第一次对话：用户介绍自己"""
    print("=" * 60)
    print("【会话 1】用户介绍自己")
    print("=" * 60)

    agent = MemoryEnabledAgent()

    convos = [
        "你好，我是李明，一个有6年经验的 Python 后端工程师",
        "我最近在做一个 AI Agent 项目，叫 Mnemis，目标是开源并商业化",
        "我比较喜欢直接给代码示例，不需要太多铺垫解释",
        "对了，我在用 Qdrant 做向量数据库，遇到了一些性能问题",
    ]

    for msg in convos:
        print(f"\n用户: {msg}")
        resp = agent.chat(msg)
        print(f"Mnemis: {resp[:150]}...")

    # 结束会话，触发语义记忆提炼
    result = agent.end_session()
    print(f"\n📊 会话结束统计：")
    print(f"  提炼了 {result['new_memories']} 条语义记忆：")
    for mem_type, content in result["memory_details"]:
        print(f"  [{mem_type}] {content}")


def show_semantic_memories():
    """查看当前所有语义记忆"""
    print("\n" + "=" * 60)
    print("📧 当前语义记忆库")
    print("=" * 60)

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)

    stats = semantic.get_stats()
    print(f"总记忆数：{stats['total']}")
    print(f"平均置信度：{stats['avg_confidence']}")
    print(f"按类型分布：{stats['by_type']}\n")

    for mem_type in MemoryType:
        memories = semantic.get_all_by_type(memory_type=mem_type, limit=5)
        if memories:
            print(f"【{mem_type.value}】")
            for m in memories:
                bar = "█" * int(m.confidence * 10)
                print(f"  {bar} {m.confidence:.1f} | {m.content}")
            print()


def session_2_verify_memory():
    """第二次会话：验证记忆是否被正确使用"""
    print("=" * 60)
    print("【会话 2】验证记忆（新实例，模拟重启）")
    print("=" * 60)

    agent = MemoryEnabledAgent()

    questions = [
        "你记得我是做什么的吗？",
        "我正在做的项目叫什么？目标是什么？",
        "给我解释一下 Qdrant 的 HNSW 索引",  # 看看会不会直接给代码示例（根据偏好）
    ]

    for q in questions:
        print(f"\n用户: {q}")
        resp = agent.chat(q)
        print(f"Mnemis: {resp[:300]}...")

    agent.end_session()


def semantic_search_demo():
    """语义检索演示：按相关话题找记忆"""
    print("\n" + "=" * 60)
    print("🔍 语义记忆检索演示")
    print("=" * 60)

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)

    queries = [
        ("向量数据库", None),
        ("用户喜欢什么风格", MemoryType.PREFERENCE),
        ("项目目标", MemoryType.GOAL),
    ]

    for query, mem_type in queries:
        results = semantic.search_by_semantic(query, top_k=3, memory_type=mem_type)
        filter_str = f"（类型过滤：{mem_type.value}）" if mem_type else ""
        print(f"\n查询：「{query}」{filter_str}")
        if results:
            for r in results:
                print(f"  [{r.memory_type.value}] {r.content}")
        else:
            print("  无相关记忆")


if __name__ == "__main__":
    session_1_introduce()
    show_semantic_memories()
    session_2_verify_memory()
    semantic_search_demo()
