from core.tools.base import ToolSchema, ToolResult
from core.tools.registry import tool
from core.knowledge_base import KnowledgeBase
from core.retriever import HybridSearcher, format_context_with_citations
from utils.logger import get_logger


logger = get_logger(__name__)


# ── 全局实例（懒加载）────────────────────────────────────────────

_kb: KnowledgeBase | None = None
_searcher: HybridSearcher | None = None


def get_searcher() -> HybridSearcher:
    """懒加载：第一次调用时初始化"""
    global _kb, _searcher
    if _searcher is None:
        logger.info("Initializing knowledge base and hybrid seacher...")
        _kb = KnowledgeBase()
        _searcher = HybridSearcher(_kb)
    return _searcher


def get_kb() -> KnowledgeBase:
    get_searcher()  # 确保 _kb 已初始化
    return _kb  # pyright: ignore[reportReturnType]


# ── Tool Schema 定义 ──────────────────────────────────────────────
SEARCH_KB_SCHEMA = ToolSchema(
    name="search_knowledge_base",
    description="""在私有知识库中搜索已存储的文档和内容
        适合使用的场景：
        - 用户询问关于已添加到知识库的文档内容
        - 用户提到「根据我的文档」「我上传的资料」等
        - 问题涉及内部资料、个人笔记、私有信息
        - 需要引用特定文档中的具体内容

        不适合使用的场景：
        - 需要实时信息（最新新闻、当前价格）→ 用 web_search
        - 通用知识问题（知识库里不太可能有）→ 直接回答
        - 用户没有提到任何文档或笔记
    """,
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询，描述你要找的内容。支持自然语言和关键词混合。",
            },
            "source_filter": {
                "type": "string",
                "description": "（可选）限制只在特定文档里搜索，填入文件名如 'report.pdf'。不填则搜索全部文档。",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 5，最多 10",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)

ADD_DOCUMENT_SCHEMA = ToolSchema(
    name="add_to_knowledge_base",
    description="""将文本内容添加到 Mnemis 的私有知识库中，以便后续检索。
        使用场景：
        - 用户说「记住这个」「把这个加入知识库」「保存这段内容」
        - 用户粘贴了一段内容并希望能以后查询
        - 用户希望 Agent 记住某个重要信息""",
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "要添加到知识库的文本内容"},
            "source_name": {
                "type": "string",
                "description": "这段内容的来源标识，如 'meeting_notes_2025' 或 'user_manual'",
            },
        },
        "required": ["content", "source_name"],
    },
)

LIST_SOURCES_SCHEMA = ToolSchema(
    name="list_knowledge_base_sources",
    description="""列出知识库中所有已存储的文档来源。
        当用户问「知识库里有什么」「我上传了哪些文件」时使用。""",
    input_schema={"type": "object", "properties": {}, "required": []},
)

# ── 工具执行函数 ──────────────────────────────────────────────────


@tool(schema=SEARCH_KB_SCHEMA)
def search_knowledge_base(
    query: str, source_filter: str | None = None, top_k: int = 5
) -> ToolResult:
    """在知识库中进行 Hybrid Search 并返回带引用的结果"""
    searcher = get_searcher()

    top_k = min(max(top_k, 20), 1)
    results = searcher.search(query=query, top_k=top_k, source_filter=source_filter)

    if not results:
        return ToolResult(
            tool_use_id="",
            content=f"知识库中未找到与 {query} 相关的内容"
            f"知识库现有文档为：{get_kb().list_sources()}",
        )

    context_str, citations = format_context_with_citations(results)

    # 把引用信息也附在结果里，让模型知道来源
    citation_summary = "\n".join(
        f"[{c['index']}] {c['source']}"
        + (f" 第{c['page_num']}页" if c["page_num"] else "")
        + f" (相关度: {c['score']:.4f})"
        for c in citations
    )

    return ToolResult(
        tool_use_id="", content=f"{context_str}\n\n---\引用来源：\n{citation_summary}"
    )


@tool(schema=ADD_DOCUMENT_SCHEMA)
def add_to_knowledge_base(content: str, source_name: str) -> ToolResult:
    """向知识库动态添加文本内容"""
    kb = get_kb()

    if not content.strip():
        return ToolResult(
            tool_use_id="", content="添加失败：内容不能为空", is_error=True
        )

    chunk_count = kb.add_text(text=content, source=source_name)

    # 添加后重建 BM25 索引
    get_searcher().rebuild_index()

    return ToolResult(
        tool_use_id="",
        content=f"✓ 已将内容添加到知识库\n"
        f"  来源标识：{source_name}\n"
        f"  切割为 {chunk_count} 个片段\n"
        f"  后续可用「查询知识库」工具检索此内容",
    )


@tool(schema=LIST_SOURCES_SCHEMA)
def list_knowledge_base_sources() -> ToolResult:
    """列出所有文档来源"""
    kb = get_kb()
    sources = kb.list_sources()
    stats = kb.get_stats()

    if not sources:
        return ToolResult(
            tool_use_id="",
            content="知识库当前为空，尚未添加任何文档。"
        )

    source_list = "\n".join(f"  • {s}" for s in sources)
    return ToolResult(
        tool_use_id="",
        content=f"知识库统计：共 {stats['total_points']} 个片段\n\n"
                f"文档来源：\n{source_list}"
    )