from core.client import LLMClient
from core.models import Message
from utils.logger import get_logger


logger = get_logger(__name__)

SYSTEM_PROMPT = """你是Mnemis. 一个正在被构建的AI研究助手.
你的风格：简洁、精准、极具洞察力
当用户的问题模糊时，先澄清再回答"""


COMMANDS = {
    "/quit": "退出对话",
    "/persona": "切换角色",
    "/clear": "清空对话历史",
    "/history": "查看当前对话历史",
    "/tokens": "查看本次会话总 token 消耗",
    "/help": "显示帮助",
}


def print_help():
    print("\n可用命令:")
    for cmd, desc in COMMANDS.items():
        print(f" {cmd:<12} {desc}")
    print()


def print_history(history: list[Message]):
    if not history:
        print("\n (对话历史为空)")
        return
    print(f"\n--对话历史（共{len(history)}条）--")
    for i, msg in enumerate(history, 1):
        role_label = "你" if msg.role == "user" else "Mnemis"
        content = msg.content if len(msg.content) < 50 else msg.content[:50] + "..."

        print(f" [{i}] {role_label}: {content}")
    print()


def run():
    client = LLMClient()
    history: list[Message] = []
    session_input_tokens = 0
    session_output_tokens = 0
    session_cost_estimate = 0

    print("=" * 50)
    print("  Mnemis CLI  —  P0 阶段实操")
    print("=" * 50)
    print("输入 /help 查看可用命令\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            if user_input == "/quit":
                print(f"会话结束\n")
                print(
                    f"\n本次会话 token 消耗：in_tokens {session_input_tokens} / out_tokens {session_output_tokens}"
                )
                print(f"估算成本：￥{session_cost_estimate:.4f}\n")
                break
            elif user_input == "/clear":
                history.clear()
                print("对话历史已清空")
            elif user_input == "/history":
                print_history(history)
            elif user_input == "/tokens":
                print(
                    f"\n本次会话 token 消耗：in_tokens {session_input_tokens} / out_tokens {session_output_tokens}"
                )
                print(f"估算成本：￥{session_cost_estimate:.4f}\n")
            elif user_input == "/help":
                print_help()
            else:
                print(f"未知命令：{user_input}，输入 /help 查看可用命令\n")
            continue

        history.append(Message(role="user", content=user_input))

        print("Mnemis: ", end="")
        response = client.chat(
            messages=history,
            system=SYSTEM_PROMPT,
            stream=True,
        )

        history.append(Message(role="assistant", content=response.content))
        session_input_tokens += response.input_tokens
        session_output_tokens += response.output_tokens
        session_cost_estimate += response.cost_estimate

        # 上下文长度预警
        estimated_tokens = session_input_tokens + session_output_tokens
        if estimated_tokens > 50_000:
            print(
                f"\n⚠️  对话已消耗约 {estimated_tokens} tokens，建议使用 /clear 清空历史\n"
            )


if __name__ == "__main__":
    run()
