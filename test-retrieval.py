# retrieval_demo.py
import core.tools
from core.memory.manager import MemoryEnabledAgent
from core.memory.database import MemoryDatabase
from core.memory.semantic import SemanticMemory
from core.memory.retriever import MemoryRetriever
from core.memory.episodic import EpisodicMemory
from core.client import LLMClient

def demo_retrieval_quality():
    """演示记忆检索的相关性筛选"""
    print("=" * 60)
    print("记忆检索质量演示")
    print("=" * 60)

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)
    episodic = EpisodicMemory(db, llm)
    retriever = MemoryRetriever(db, semantic, episodic)

    # 用不同查询测试检索结果
    queries = [
        "Qdrant 向量数据库性能调优",
        "用户的编程偏好和工作风格",
        "项目开源和商业化计划",
    ]

    for q in queries:
        print(f"\n查询：「{q}」")
        result = retriever.retrieve_for_context(q)
        print(f"  语义记忆：{len(result['semantic_memories'])} 条")
        for m in result['semantic_memories'][:3]:
            print(f"    [{m.memory_type.value}] {m.content[:50]}")
        print(f"  相关历史：{len(result['episodic_summaries'])} 条")
        print(f"  预估消耗：~{result['total_token_estimate']} tokens")

        ctx = retriever.format_memory_context(result)
        print(f"  注入字符数：{len(ctx)}")


def demo_active_recall():
    """演示 Agent 主动使用 recall_memories 工具"""
    print("\n" + "=" * 60)
    print("主动记忆召回演示")
    print("=" * 60)

    agent = MemoryEnabledAgent()

    # 这些问题应该触发 Agent 主动调用 recall_memories 工具
    memory_questions = [
        "你还记得我说过我在做什么项目吗？",
        "根据你对我的了解，给我推荐一个合适的学习路径",
        "给我看看你目前对我的完整记录",
    ]

    for q in memory_questions:
        print(f"\n用户: {q}")
        resp = agent.chat(q)
        print(f"Mnemis: {resp[:400]}...")

    agent.end_session()


def demo_system_prompt_inspection():
    """直接打印 system prompt，验证记忆注入效果"""
    print("\n" + "=" * 60)
    print("System Prompt 记忆注入检查")
    print("=" * 60)

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)
    episodic = EpisodicMemory(db, llm)
    retriever = MemoryRetriever(db, semantic, episodic)

    test_query = "Python 异步编程"
    result = retriever.retrieve_for_context(test_query)
    memory_ctx = retriever.format_memory_context(result, max_tokens=600)

    print(f"查询：「{test_query}」")
    print(f"注入的记忆上下文（共 {len(memory_ctx)} 字符）：")
    print("─" * 40)
    print(memory_ctx if memory_ctx else "（无记忆可注入）")
    print("─" * 40)
    print(f"\n预估额外 token 消耗：~{result['total_token_estimate']}")


if __name__ == "__main__":
    demo_retrieval_quality()
    demo_active_recall()
    demo_system_prompt_inspection()