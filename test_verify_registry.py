# verify_registry.py
from core.client import LLMClient
import core.tools  # 触发所有工具注册
from core.tools.registry import get_registry


def verify():
    registry = get_registry()
    client = LLMClient()

    messages, tool_calls = client.chat_with_tools(
        messages=[{"role": "user", "content": "新建a.txt文件，写入内容：hello, world"}],
        tools=registry.get_schemas(),
        system="按用户要求完成任务",
    )

    if tool_calls:
        for tc in tool_calls:
            result = registry.execute(tc.name,tc.input)



if __name__ == "__main__":
    verify()
