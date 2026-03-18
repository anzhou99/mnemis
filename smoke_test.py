# smoke_test.py — 验证整个调用链路是否通畅
from core.client import LLMClient
from core.models import Message


def main():
    client = LLMClient()

    # 测试1：单轮对话
    print("=== 测试1：单轮对话 ===")
    response = client.chat(
        messages=[Message(role="user", content="用一句话解释什么是 AI Agent")],
        system="你是一个简洁的技术助手，回答必须在30字以内。",
        temperature=0.3,
    )
    print(f"回答：{response.content}")

    # 测试2：Streaming
    print("=== 测试2：Streaming 输出 ===")
    client.chat(
        messages=[Message(role="user", content="用三句话介绍 Python")],
        system="你是一个编程老师。",
        stream=True,
    )


if __name__ == "__main__":
    main()
