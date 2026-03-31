from tavily import TavilyClient
from urllib3 import response
from core.tools.base import ToolSchema, ToolResult
from core.config import settings
from utils.logger import get_logger


logger = get_logger(__name__)

# ── Schema 定义（告诉模型这个工具是什么、怎么用）──────────────────

WEB_SEARCH_SCHEMA = ToolSchema(
    name="web_search",
    description="""搜索互联网获取实时或最新信息
   适用于:
    - 需要今日价格、最新新闻、近期事件（2024年后的信息）
    - 需要验证可能已过时的事实
    - 需要特定人物、产品、地点的详细信息
   不适用:
    - 通用知识、数学计算、代码编写等不依赖时效性的问题
    """,
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，建议精确具体，可加时间限定（如'2025年'）",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果数量，默认3，最多5",
                "default": 3,
            },
        },
        "required": ["query"],
    },
)


# ── 执行函数（真正去调 Tavily API）────────────────────────────────


def execute_web_search(query: str, max_results: int = 3) -> ToolResult:
    """
    执行网络搜索。
    注意：这个函数不接收 ToolCall 对象，只接收解包后的参数。
    这样工具函数本身与 Agent 框架解耦，可以单独测试。
    """
    logger.debug(f"Searching: '{query}' (max_results={max_results})")

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query, max_results=max_results, include_answer=True
        )

        # 格式化结果
        parts = []
        if response.get("answer"):
            parts.append(f"摘要：{response['answer']}\n")

        for i, result in enumerate(response.get("results", []), 1):
            parts.append(
                f"[{i}] {result['title']}\n"
                f"来源：{result['url']}\n"
                f"内容：{result['content'][:300]}..."  # 截断避免占用过多 token
            )

        content = "\n\n".join(parts) if parts else "未找到相关内容"

        logger.debug(f"Search done: {len(response.get('results', []))} results")
        return ToolResult(tool_use_id="", content=content)  # id 由调用者填入

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return ToolResult(tool_use_id="", content=f"搜索失败: {str(e)}", is_error=True)
