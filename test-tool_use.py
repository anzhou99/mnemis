from core.client import LLMClient
from core.tools.registry import get_registry
import core.tools


SYSTEM = """你是 Mnemis，一个 AI 研究助手。
你有网络搜索和文件读写能力。需要最新信息时主动搜索，不要靠猜测。"""


def run_single_tool_call(user_question: str):
    """演示单次工具调用的完整流程"""
    registry = get_registry()
    client = LLMClient()
    messages = [{"role": "user", "content": user_question}]

    print(f"\n用户：{user_question}")
    print("-" * 50)

    messages, tool_calls = client.chat_with_tools(
        messages=messages, tools=registry.get_schemas(), system=SYSTEM
    )

    if not tool_calls:
        final_text = next(
            (b.text for b in messages[-1]["content"] if hasattr(b, "text"))
        )

        print(f"Mnemis（直接回答）：{final_text}")
        return

    # 执行所有工具调用
    tool_results = []
    for tc in tool_calls:
        print(f"\n🔧 调用工具：{tc.name}")
        print(f"   参数：{tc.input}")

        result = registry.execute(tc.name, tc.input)
        result.tool_use_id = tc.id

        print(f"   结果：{result.content[:150]}...")
        tool_results.append(result)

    messages = client.append_tool_results(messages, tool_results)
    print("*" * 10)
    print(messages)
    messages, _ = client.chat_with_tools(
        messages=messages, tools=registry.get_schemas(), system=SYSTEM
    )

    final_text = next(
        (b.text for b in messages[-1]["content"] if hasattr(b, "text")), ""
    )

    print(f"\nMnemis：{final_text}")


if __name__ == "__main__":
    # 测试1：需要搜索的问题
    run_single_tool_call("今天以太坊的价格大概是多少？")

    # 测试2：不需要工具的问题（看模型会不会乱用工具）
    # run_single_tool_call("用 Python 写一个快速排序函数")
