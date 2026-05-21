# verify_memory_db.py
from core.memory.database import MemoryDatabase
from core.memory.models import Memory, MemoryType
import uuid
from datetime import datetime

def verify():
    db = MemoryDatabase()

    print("=== 验证 Session 操作 ===")
    session = db.create_session()
    print(f"✓ 创建会话：{session.id}")

    db.save_message(session.id, "user", "我是一个 Python 开发者")
    db.save_message(session.id, "assistant", "了解！我会记住这一点。")
    msgs = db.get_session_messages(session.id)
    print(f"✓ 保存并读取消息：{len(msgs)} 条")

    db.end_session(session.id, "用户介绍", "用户表示自己是 Python 开发者")
    recent = db.get_recent_sessions(limit=3)
    print(f"✓ 最近会话数：{len(recent)}")

    print("\n=== 验证 Memory 操作 ===")
    mem = Memory(
        id=str(uuid.uuid4()),
        content="用户是 Python 开发者，偏好简洁的技术回答",
        memory_type=MemoryType.FACT,
        confidence=0.9,
        source_session_id=session.id,
    )
    db.save_memory(mem)
    print(f"✓ 保存记忆：{mem.content[:40]}...")

    memories = db.get_memories_by_type(MemoryType.FACT)
    print(f"✓ 按类型检索：找到 {len(memories)} 条 FACT 类记忆")

    all_memories = db.get_memories_by_type()
    print(f"✓ 全量检索：共 {len(all_memories)} 条记忆")

    print(f"\n=== 数据库统计 ===")
    stats = db.get_stats()
    print(f"  会话数：{stats['total_sessions']}")
    print(f"  记忆数：{stats['total_memories']}")
    print(f"  按类型：{stats['by_type']}")

    print("\n✅ 所有验证通过")

if __name__ == "__main__":
    verify()