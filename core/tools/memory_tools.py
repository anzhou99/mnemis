# core/tools/memory_tools.py
from core.tools.base import ToolSchema, ToolResult
from core.tools.registry import tool
from core.memory.models import MemoryType
from utils.logger import get_logger

logger = get_logger(__name__)

# ── 懒加载全局实例 ────────────────────────────────────────────────

_retriever = None

def get_retriever():
    """懒加载记忆检索器（避免循环导入）"""
    global _retriever
    if _retriever is None:
        from core.memory.database import MemoryDatabase
        from core.memory.semantic import SemanticMemory
        from core.memory.episodic import EpisodicMemory
        from core.memory.retriever import MemoryRetriever
        from core.client import LLMClient

        db = MemoryDatabase()
        llm = LLMClient()
        semantic = SemanticMemory(db, llm)
        episodic = EpisodicMemory(db, llm)
        _retriever = MemoryRetriever(db, semantic, episodic)
    return _retriever


# ── Tool Schema 定义 ──────────────────────────────────────────────

RECALL_MEMORIES_SCHEMA = ToolSchema(
    name="recall_memories",
    description="""主动检索关于用户的长期记忆。

适合使用：
- 用户提到「你还记得...」「我之前说过...」「上次我们讨论的...」
- 需要了解用户在某个话题上的背景偏好
- 用户问「你对我了解多少」

不适合使用：
- 当前对话里已经有明确信息，不需要回忆
- 查询知识库文档（用 search_knowledge_base）""",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要检索的记忆主题，如「用户的技术背景」「用户偏好」"
            },
            "memory_type": {
                "type": "string",
                "description": "可选，过滤记忆类型：preference / fact / background / goal",
                "enum": ["preference", "fact", "background", "goal"]
            }
        },
        "required": ["query"]
    }
)

SUMMARIZE_USER_PROFILE_SCHEMA = ToolSchema(
    name="summarize_user_profile",
    description="""生成关于用户的完整画像摘要。
当用户问「你了解我多少」「给我看看你对我的记录」时使用。""",
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    }
)


# ── 执行函数 ──────────────────────────────────────────────────────

@tool(schema=RECALL_MEMORIES_SCHEMA)
def recall_memories(
    query: str,
    memory_type: str | None = None,
) -> ToolResult:
    """主动检索用户记忆"""
    retriever = get_retriever()

    mem_type = MemoryType(memory_type) if memory_type else None

    # 语义检索
    memories = retriever.semantic.search_by_semantic(
        query=query,
        top_k=8,
        memory_type=mem_type,
    )

    if not memories:
        return ToolResult(
            tool_use_id="",
            content=f"未找到关于「{query}」的相关记忆。"
        )

    lines = []
    for m in memories:
        lines.append(
            f"[{m.memory_type.value}] "
            f"(置信度 {m.confidence:.1f}) "
            f"{m.content}"
        )

    return ToolResult(
        tool_use_id="",
        content=f"关于「{query}」的记忆（共 {len(memories)} 条）：\n\n"
                + "\n".join(lines)
    )


@tool(schema=SUMMARIZE_USER_PROFILE_SCHEMA)
def summarize_user_profile() -> ToolResult:
    """生成用户完整画像"""
    retriever = get_retriever()

    all_memories = retriever.semantic.get_all_by_type(
        min_confidence=0.3,
        limit=50,
    )

    if not all_memories:
        return ToolResult(
            tool_use_id="",
            content="还没有关于你的记忆记录。和我多聊聊，我会逐渐了解你！"
        )

    # 按类型分组展示
    grouped: dict[str, list] = {}
    for m in all_memories:
        grouped.setdefault(m.memory_type.value, []).append(m)

    type_labels = {
        "preference": "偏好与习惯",
        "fact":       "基本信息",
        "background": "背景与经历",
        "goal":       "目标与计划",
    }

    parts = [f"📋 关于你的记忆（共 {len(all_memories)} 条）\n"]
    for t, label in type_labels.items():
        if t in grouped:
            items = "\n".join(
                f"  • {m.content} (置信度 {m.confidence:.1f})"
                for m in grouped[t]
            )
            parts.append(f"【{label}】\n{items}")

    # 最近会话历史
    sessions = retriever.db.get_recent_sessions(limit=5)
    if sessions:
        session_lines = []
        for s in sessions:
            if s.title:
                session_lines.append(
                    f"  • {s.started_at.strftime('%m-%d')} {s.title}"
                )
        if session_lines:
            parts.append("【近期对话】\n" + "\n".join(session_lines))

    return ToolResult(
        tool_use_id="",
        content="\n\n".join(parts)
    )