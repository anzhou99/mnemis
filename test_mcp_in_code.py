import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def demo_mcp_client():
    """在 Python 代码里通过 MCP 协议调用 Mnemis Server"""

    server_params = StdioServerParameters(
        command="uv", args=["run", "python", "mcp_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化握手
            await session.initialize()

            # 列出所有可用工具
            tools = await session.list_tools()
            print(f"可用工具：{[t.name for t in tools.tools]}\n")

            # 列出所有资源
            resources = await session.list_resources()
            print(f"可用资源：{[r.uri for r in resources.resources]}\n")

            # 调用工具：搜索知识库
            print("调用 search_knowledge_base...")
            result = await session.call_tool(
                name="search_knowledge_base",
                arguments={"query": "Python 异步编程", "top_k": 3},
            )
            print(f"搜索结果：{result.content[0].text}...\n")

            # 读取资源：知识库文档列表
            print("读取 mnemis://kb/sources...")
            resource = await session.read_resource("mnemis://kb/sources")
            print(f"资源内容：{resource.contents[0].text}...\n")

            # 获取提示词模板
            print("获取 research_topic 提示词...")
            prompt = await session.get_prompt(
                "research_topic", {"topic": "asyncio 事件循环", "depth": "quick"}
            )
            print(f"提示词：{prompt.messages[0].content.text}...")


if __name__ == "__main__":
    asyncio.run(demo_mcp_client())
