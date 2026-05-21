# api/routers/memory.py
from fastapi import APIRouter, Depends
from api.dependencies import get_current_user

router = APIRouter(prefix="/memory", tags=["记忆"])


@router.get("/profile")
def get_user_profile(current_user: dict = Depends(get_current_user)):
    """获取 Agent 对当前用户的完整记忆画像"""
    from core.memory.database import MemoryDatabase
    from core.memory.semantic import SemanticMemory
    from core.client import LLMClient

    db = MemoryDatabase()
    semantic = SemanticMemory(db, LLMClient())
    memories = semantic.get_all_by_type(limit=50)

    return {
        "total": len(memories),
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "type": m.memory_type.value,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ],
    }


@router.get("/sessions")
def get_recent_sessions(
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
):
    """获取最近的对话会话列表"""
    from core.memory.database import MemoryDatabase
    db = MemoryDatabase()
    sessions = db.get_recent_sessions(limit=limit)
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "summary": s.summary,
                "message_count": s.message_count,
                "started_at": s.started_at.isoformat(),
            }
            for s in sessions
        ]
    }