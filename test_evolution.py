# evolution_demo.py
import core.tools
from core.memory.manager import MemoryEnabledAgent
from core.memory.database import MemoryDatabase
from core.memory.semantic import SemanticMemory
from core.memory.conflict import ConflictResolver, MemoryEvolution
from core.memory.models import MemoryType
from core.client import LLMClient
import time


def demo_conflict_detection():
    """演示冲突检测：用户改变了偏好"""
    print("=" * 60)
    print("演示1：冲突检测——用户改变了回答风格偏好")
    print("=" * 60)

    agent = MemoryEnabledAgent()

    # 第一次会话：说喜欢详细解释
    print("\n【会话 A】用户说喜欢详细解释")
    agent.chat("你好，我喜欢详细的解释，不怕长，越详细越好")
    result_a = agent.end_session()
    print(f"提炼记忆：{result_a['memory_details']}")

    time.sleep(1)

    # 第二次会话：改变了偏好
    print("\n【会话 B】用户改变了偏好")
    agent.chat("我发现还是简洁的回答更适合我，之前说的详细解释我觉得太冗长了")
    result_b = agent.end_session()
    print(f"提炼记忆：{result_b['memory_details']}")

    # 检查冲突处理结果
    print("\n📊 冲突处理后的记忆库：")
    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)
    prefs = semantic.get_all_by_type(MemoryType.PREFERENCE)
    for p in prefs:
        status = "✓ 有效" if p.confidence > 0.3 else "✗ 已废弃"
        print(f"  {status} (conf={p.confidence:.2f}) {p.content}")

    # 查看冲突日志
    with db._conn() as conn:
        conflicts = conn.execute(
            "SELECT * FROM memory_conflicts ORDER BY resolved_at DESC LIMIT 5"
        ).fetchall()
    if conflicts:
        print("\n🔍 冲突日志：")
        for c in conflicts:
            print(f"  {c['conflict_type']} → {c['resolution']}")


def demo_time_decay():
    """演示时间衰减（用模拟数据）"""
    print("\n" + "=" * 60)
    print("演示2：时间衰减效果")
    print("=" * 60)

    import uuid
    from datetime import datetime, timedelta
    from core.memory.models import Memory

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)
    evolution = MemoryEvolution(db, semantic)

    # 手动写入一些「老」记忆（模拟过去的数据）
    old_memories = [
        ("用户正在研究 LangChain 框架", MemoryType.BACKGROUND, 120),  # 120天前
        ("用户计划参加 AI 黑客马拉松", MemoryType.GOAL, 90),           # 90天前
        ("用户偏好简洁回答", MemoryType.PREFERENCE, 10),               # 10天前（新鲜）
    ]

    inserted = []
    for content, mem_type, days_ago in old_memories:
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=mem_type,
            confidence=1.0,
            created_at=datetime.now() - timedelta(days=days_ago),
        )
        db.save_memory(mem)
        inserted.append((content, days_ago, mem.confidence))
        print(f"插入（{days_ago}天前）：{content}")

    print("\n执行时间衰减...")
    stats = evolution.apply_time_decay()
    print(f"衰减结果：{stats}")

    print("\n衰减后的记忆状态：")
    for content, days_ago, _ in inserted:
        mems = db.get_memories_by_type(min_confidence=0.0, limit=100)
        for m in mems:
            if m.content == content:
                print(f"  [{days_ago}天前] conf: 1.0 → {m.confidence:.3f} | {content[:40]}")
                break


def demo_consolidation():
    """演示记忆整合：合并重复记忆"""
    print("\n" + "=" * 60)
    print("演示3：记忆整合——合并重复记忆")
    print("=" * 60)

    import uuid
    from core.memory.models import Memory

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)
    evolution = MemoryEvolution(db, semantic)

    # 插入一些重复的偏好记忆
    duplicates = [
        "用户喜欢看代码示例",
        "用户偏好代码而非文字解释",
        "用户希望回答中有可运行的代码",
    ]
    for content in duplicates:
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=MemoryType.PREFERENCE,
            confidence=0.9,
        )
        db.save_memory(mem)
        print(f"插入重复记忆：{content}")

    print("\n执行记忆整合...")
    merged_count = evolution.consolidate_memories(llm)
    print(f"整合结果：合并了 {merged_count} 条记忆")

    print("\n整合后的偏好记忆：")
    remaining = semantic.get_all_by_type(MemoryType.PREFERENCE)
    for m in remaining:
        print(f"  • {m.content}")


if __name__ == "__main__":
    demo_conflict_detection()
    demo_time_decay()
    demo_consolidation()