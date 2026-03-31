from pydantic import BaseModel, Field
from typing import Literal
from core.client import LLMClient
from core.models import Message

client = LLMClient()

# -----------------
# 例子1：任务分析
# -----------------
class TaskAnalysis(BaseModel):
    summary: str = Field(description="用一句话概括任务")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="难度评估")
    steps: list[str] = Field(description="完成任务的步骤列表")
    estimated_hours: float = Field(description="预估工时（小时）")


def demo_task_analysis():
    print("\n=== 例子1：任务分析 ===")

    result = client.structured_chat(
        messages=[
            Message(
                role="user",
                content="我需要开发一个支持多轮对话的 CLI 聊天机器人，用 Python 实现",
            )
        ],
        response_model=TaskAnalysis,
        system="你是一个经验丰富的技术项目评估专家。",
    )

    print(f"摘要：{result.summary}")
    print(f"难度：{result.difficulty}")
    print(f"预估工时：{result.estimated_hours}")
    print("步骤：")
    for i, step in enumerate(result.steps, 1):
        print(f" {i}.  {step}")


# -----------------
# 例子2：情感分析
# -----------------
class SentimentScore(BaseModel):
    label: Literal["positive", "neutral", "negative"] = Field(description="情感标签")
    confidence: float = Field(description="置信度 0-1 之间", ge=0, le=1)
    reasoning: str = Field(description="判断依据，用一句话说明")


def demo_sentiment():
    print("\n=== 例子2：情感分析 ===")

    texts = [
        "这个产品真的太棒了，完全超出预期！",
        "一般般吧，没什么特别的。",
        "用了三天就坏了，客服还不理人，垃圾产品！",
    ]

    for text in texts:
        result = client.structured_chat(
            messages=[Message(role="user", content=f"分析如下语句的情感：{text}")],
            response_model=SentimentScore,
        )
        bar = "█" * int(result.confidence * 10) + "░" * (
            10 - int(result.confidence * 10)
        )
        print(f"\n  文本：{text}")
        print(f"  情感：{result.label} [{bar}] {result.confidence:.0%}")
        print(f"  理由：{result.reasoning}")


# -----------------
# 例子3：Agent 决策
# -----------------
class AgentAction(BaseModel):
    thought: str = Field(description="Agent 的推理过程")
    action: Literal["search", "read_file", "write_file", "ask_user", "finish"] = Field(
        description="要执行的动作类型"
    )
    action_input: str = Field(description="动作的输入参数")
    is_final: bool = Field(description="这是否是最后一步")


def demo_agent_decision():
    print("\n=== 例子 3：Agent 决策（Tool Use 预演）===")

    task = "帮我研究一下 Python 异步编程的最佳实践，整理成笔记保存"

    result = client.structured_chat(
        messages=[
            Message(role="user", content=f"任务：{task}\n你的下一步行动是什么？")
        ],
        response_model=AgentAction,
        system="""你是一个 AI Agent，负责规划并执行任务。
你有以下工具可以使用：
- search：搜索网络信息
- read_file：读取本地文件
- write_file：写入文件
- ask_user：向用户询问信息
- finish：任务完成

每次只输出下一步行动。""",
    )

    print(f"  思考：{result.thought}")
    print(f"  动作：{result.action}")
    print(f"  输入：{result.action_input}")
    print(f"  是否完成：{result.is_final}")
    # print("\n  ↑ 这就是 ReAct Agent 循环的核心，未来会深入探讨这块内容")


# -----------------
# 例子4：错误重试
# -----------------
def demo_error_recovery():
    print("\n=== 实验：错误恢复机制 ===")

    class StrictModel(BaseModel):
        name: str = Field(description="名称")
        count: int = Field(description="模块数量，必须是整数")
        tags: list[str] = Field(description="标签列表")

    # 故意给一个容易让 LLM 犯错的 prompt
    result = client.structured_chat(
        messages=[Message(
            role="user",
            content="产品名称：Mnemis，大约有三点几个功能模块，标签包括 AI、Agent、Python，当你输出count时，你必须输出中文的'三点二'"
        )],
        system="根据用户描述提前必要信息",
        response_model=StrictModel,
        max_retries=3,
    )
    print(f"  解析成功：{result}")


if __name__ == "__main__":
    # demo_task_analysis()
    # demo_sentiment()
    # demo_agent_decision()
    demo_error_recovery()
