"""
Mnemis MCP Server
把知识库检索、记忆查询等能力标准化地暴露给任意 MCP 客户端。

运行方式：
  uv run python mcp_server.py

在 Claude Desktop 配置：
  {
    "mcpServers": {
      "mnemis": {
        "command": "uv",
        "args": ["run", "python", "/path/to/mnemis/mcp_server.py"]
      }
    }
  }
"""

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


# 延迟导入 Mnemis 核心模块（避免启动时报错影响 MCP 握手）
_kb = None
_semantic = None
_db = None


def get_kb():
    global _kb
    if _kb is None:
        from core.knowledge_base import KnowledgeBase

        _kb = KnowledgeBase()
    return _kb


def get_semantic():
    global _semantic, _db
    if _semantic is None:
        from core.memory.database import MemoryDatabase
        from core.memory.semantic import SemanticMemory
        from core.client import LLMClient

        _db = MemoryDatabase()
        _semantic = SemanticMemory(_db, LLMClient())
    return _semantic


def get_db():
    get_semantic()  # 确保 _db 已初始化
    return _db


# ── 创建 MCP Server 实例 ──────────────────────────────────────────
app = Server("mnemis")


# ═══════════════════════════════════════════════════════════════════
# TOOLS（工具）：模型可以调用执行的函数
# ═══════════════════════════════════════════════════════════════════


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """声明所有可用工具"""
    return [
        types.Tool(
            name="search_knowledge_base",
            description="""在 Mnemis 私有知识库中搜索文档内容。
                    使用 Hybrid Search（向量 + BM25）确保语义和关键词双重覆盖。
                    适合：查找已添加到知识库的文档内容、私有笔记、上传的资料。""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，支持自然语言和关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量（1-10，默认5）",
                        "default": 5,
                    },
                    "source_filter": {
                        "type": "string",
                        "description": "可选，限定只在某个文档里搜索，填文件名",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="add_to_knowledge_base",
            description="将文本内容添加到 Mnemis 知识库，以便后续检索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要添加的文本内容"},
                    "source_name": {
                        "type": "string",
                        "description": "内容来源标识，如 'meeting_notes_0315'",
                    },
                },
                "required": ["content", "source_name"],
            },
        ),
        types.Tool(
            name="recall_memories",
            description="检索关于用户的长期记忆（偏好、背景、目标等）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的记忆主题"},
                    "memory_type": {
                        "type": "string",
                        "description": "过滤类型：preference / fact / background / goal",
                        "enum": ["preference", "fact", "background", "goal"],
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_user_profile",
            description="获取用户的完整画像摘要（所有已记录的偏好、事实、背景、目标）。",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """执行工具调用"""
    print(name)
    if name == "search_konwledge_base":
        return await _tool_search_kb(arguments)

    elif name == "add_to_knowledge_base":
        return await _tool_add_to_kb(arguments)

    elif name == "recall_memories":
        return await _tool_recall_memories(arguments)

    elif name == "get_user_profile":
        return await _tool_get_user_profile()

    else:
        return [types.TextContent(type="text", text=f"未知工具：{name}")]


async def _tool_search_kb(args: dict) -> list[types.TextContent]:
    try:
        from core.retriever import HybridSearcher, format_context_with_citations

        kb = get_kb()
        searcher = HybridSearcher(kb)

        results = searcher.search(
            query=args["query"],
            top_k=args.get("top_k", 5),
            source_filter=args.get("source_filter"),
        )

        if not results:
            return [
                types.TextContent(
                    type="text", text=f"知识库中未找到与「{args['query']}」相关的内容。"
                )
            ]

        context_str, citations = format_context_with_citations(results)
        citation_lines = "\n".join(
            f"[{c['index']}] {c['source']}"
            + (f" 第{c['page_num']}页" if c.get("page_num") else "")
            + f" (相关度: {c['score']:.3f})"
            for c in citations
        )

        return [
            types.TextContent(
                type="text", text=f"{context_str}\n\n---\n引用来源：\n{citation_lines}"
            )
        ]
    except Exception as e:
        return [types.TextContent(type="text", text=f"检索失败：{str(e)}")]


async def _tool_add_to_kb(args: dict) -> list[types.TextContent]:
    try:
        kb = get_kb()
        count = kb.add_text(args["content"], source=args["source_name"])
        return [
            types.TextContent(
                type="text",
                text=f"✓ 已添加到知识库\n来源：{args['source_name']}\n切割为 {count} 个片段",
            )
        ]
    except Exception as e:
        return [types.TextContent(type="text", text=f"添加失败：{str(e)}")]


async def _tool_recall_memories(args: dict) -> list[types.TextContent]:
    try:
        from core.memory.models import MemoryType

        semantic = get_semantic()

        mem_type = None
        if args.get("memory_type"):
            try:
                mem_type = MemoryType(args["memory_type"])
            except ValueError:
                pass

        memories = semantic.search_by_semantic(
            query=args["query"],
            top_k=8,
            memory_type=mem_type,
        )

        if not memories:
            return [
                types.TextContent(
                    type="text", text=f"未找到关于「{args['query']}」的相关记忆。"
                )
            ]

        lines = [f"找到 {len(memories)} 条相关记忆：\n"]
        for m in memories:
            lines.append(
                f"[{m.memory_type.value}] (置信度 {m.confidence:.1f}) {m.content}"
            )

        return [types.TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        return [types.TextContent(type="text", text=f"记忆检索失败：{str(e)}")]


async def _tool_get_user_profile() -> list[types.TextContent]:
    try:
        semantic = get_semantic()
        db = get_db()

        all_memories = semantic.get_all_by_type(min_confidence=0.3, limit=50)
        if not all_memories:
            return [types.TextContent(type="text", text="暂无用户记忆记录。")]

        from core.memory.models import MemoryType

        type_labels = {
            "preference": "偏好与习惯",
            "fact": "基本信息",
            "background": "背景与经历",
            "goal": "目标与计划",
        }

        grouped: dict[str, list] = {}
        for m in all_memories:
            grouped.setdefault(m.memory_type.value, []).append(m)

        parts = [f"用户画像（共 {len(all_memories)} 条记忆）\n"]
        for t, label in type_labels.items():
            if t in grouped:
                items = "\n".join(
                    f"  • {m.content} (置信度 {m.confidence:.1f})" for m in grouped[t]
                )
                parts.append(f"【{label}】\n{items}")

        # 最近会话
        sessions = db.get_recent_sessions(limit=5)
        if sessions:
            session_lines = [
                f"  • {s.started_at.strftime('%m-%d')} {s.title or '未命名'}"
                for s in sessions
                if s.title
            ]
            if session_lines:
                parts.append("【近期对话】\n" + "\n".join(session_lines))

        return [types.TextContent(type="text", text="\n\n".join(parts))]

    except Exception as e:
        return [types.TextContent(type="text", text=f"获取用户画像失败：{str(e)}")]


# ═══════════════════════════════════════════════════════════════════
# RESOURCES（资源）：模型可以读取的数据源
# ═══════════════════════════════════════════════════════════════════


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """声明所有可用资源"""
    return [
        types.Resource(
            uri="mnemis://kb/sources",
            name="知识库文档列表",
            description="列出知识库中所有已存储的文档来源",
            mimeType="application/json",
        ),
        types.Resource(
            uri="mnemis://memory/profile",
            name="用户画像",
            description="用户的完整记忆画像（JSON 格式）",
            mimeType="application/json",
        ),
        types.Resource(
            uri="mnemis://memory/recent-sessions",
            name="最近会话",
            description="最近 10 次对话的摘要列表",
            mimeType="application/json",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """读取资源内容"""
    if uri == "mnemis://kb/sources":
        try:
            kb = get_kb()
            sources = kb.list_sources()
            stats = kb.get_stats()
            return json.dumps(
                {
                    "total_chunks": stats["total_points"],
                    "sources": sources,
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif uri == "mnemis://memory/profile":
        try:
            semantic = get_semantic()
            memories = semantic.get_all_by_type(limit=100)
            return json.dumps(
                [
                    {
                        "content": m.content,
                        "type": m.memory_type.value,
                        "confidence": m.confidence,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in memories
                ],
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif uri == "mnemis://memory/recent-sessions":
        try:
            db = get_db()
            sessions = db.get_recent_sessions(limit=10)
            return json.dumps(
                [
                    {
                        "id": s.id,
                        "title": s.title,
                        "started_at": s.started_at.isoformat(),
                        "summary": s.summary,
                        "message_count": s.message_count,
                    }
                    for s in sessions
                ],
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"未知资源 URI: {uri}"})


# ═══════════════════════════════════════════════════════════════════
# PROMPTS（提示词模板）
# ═══════════════════════════════════════════════════════════════════


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="research_topic",
            description="深度研究某个技术主题，结合 Mnemis 知识库和网络搜索",
            arguments=[
                types.PromptArgument(
                    name="topic",
                    description="要研究的主题",
                    required=True,
                ),
                types.PromptArgument(
                    name="depth",
                    description="研究深度：quick（快速概览）/ deep（深度分析）",
                    required=False,
                ),
            ],
        )
    ]


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name == "research_topic":
        if arguments:
            topic = arguments.get("topic", "未指定主题")
            depth = arguments.get("depth", "deep")
        else:
            return types.GetPromptResult(description="未知提示词", messages=[])

        depth_instruction = (
            "请给出 300 字以内的快速概览"
            if depth == "quick"
            else "请进行深度分析，包含原理、实践案例和代码示例，约 1500 字"
        )

        return types.GetPromptResult(
            description=f"研究：{topic}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""请深入研究以下主题：{topic}
                研究步骤：
                1. 先用 search_knowledge_base 工具查询我的私有知识库，看是否有相关内容
                2. 如需补充，使用网络搜索获取最新信息
                3. {depth_instruction}
                4. 如果知识库里有相关内容，在回答中注明来源

                开始研究：""",
                    ),
                )
            ],
        )

    return types.GetPromptResult(description="未知提示词", messages=[])


# ── 启动 Server ───────────────────────────────────────────────────


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )
        print('started')


if __name__ == "__main__":
    asyncio.run(main())
