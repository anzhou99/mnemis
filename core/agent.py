from dataclasses import dataclass, field
from typing import Literal
from core.client import LLMClient
from core.tools.base import ToolResult
from core.tools.registry import ToolRegistry, get_registry
from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


input_tokens_price = 2 / 100_0000
output_tokens_price = 3 / 100_0000


@dataclass
class AgentStep:
    """记录 Agent 每一步的执行情况，用于日志和调试"""

    step_num: int
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    final_answer: str = ""


@dataclass
class AgentResult:
    """Agent 完整运行结果"""

    success: bool
    answer: str
    steps: list[AgentStep]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    stop_reason: (
        Literal["finished", "max_steps", "budget_exceeded", "error", "loop_detected"]
        | None
    ) = None

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cost_estimate_usd(self) -> float:
        return (
            self.total_input_tokens * input_tokens_price
            + self.total_output_tokens * output_tokens_price
        )  


class ReActAgent:
    """
    ReAct 推理循环的核心实现。

    工作原理：
    1. 把用户任务 + 可用工具发给模型
    2. 模型决定调用哪个工具（Action）
    3. 执行工具，把结果追加到 messages（Observation）
    4. 模型看到结果后决定下一步（Thought → 下一个 Action）
    5. 循环直到模型给出最终答案，或触发安全限制
    """

    def __init__(
        self,
        system_prompt: str | None = None,
        max_steps: int = 10,
        max_tokens_budgent: int = 100_000,
        tool_names: list[str] | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.client = LLMClient()
        self.registry = registry or get_registry()
        self.max_steps = max_steps
        self.system_prompt = system_prompt or self._default_system()
        self.max_tokens_budgent = max_tokens_budgent
        self.tool_names = tool_names

    def _default_system(self) -> str:
        return """你是 Mnemis，一个能自主完成复杂任务的 AI 研究助手。
        
## 工作方式
你可以使用工具来获取信息、读写文件。遇到需要实时信息或多步骤的任务时，
主动使用工具，不要依赖你的训练知识。

## 行为准则
- 每次只专注于当前最重要的下一步
- 工具执行失败时，分析原因并尝试替代方案
- 完成任务后，给出清晰的总结"""

    def run(self, task: str) -> AgentResult:
        """
        运行 ReAct 循环，执行给定任务。
        """
        logger.info(
            f"Agent starting | task='{task[:60]}...' | max_steps={self.max_steps}"
        )

        # 获取工具（子集或全部）
        tools = self.registry.get_schemas(self.tool_names)

        if not tools:
            logger.warning("No tools available, agent will run without tools")
        # 初始化 messages
        messages = [{"role": "user", "content": task}]

        steps: list[AgentStep] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"── Step {step_num}/{self.max_steps} ──")
            current_step = AgentStep(step_num=step_num)

            if total_input_tokens + total_output_tokens >= self.max_tokens_budgent:
                logger.warning(
                    f"Token budget exceeded: {total_input_tokens + total_output_tokens}"
                )
                steps.append(current_step)
                return AgentResult(
                    success=False,
                    answer="任务未完成：已超出 token 预算上限。",
                    steps=steps,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    stop_reason="budget_exceeded",
                )

            try:
                messages, tool_calls = self.client.chat_with_tools(
                    messages=messages, tools=tools, system=self.system_prompt
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return AgentResult(
                    success=False,
                    answer=f"LLM 调用失败：{e}",
                    steps=steps,
                    stop_reason="error",
                )

            # 累计 token 消耗（从最新的 assistant 消息里读取）
            # 注意：chat_with_tools 内部已经记录了，这里我们用一个简单估算
            # 在真实项目里你可以让 chat_with_tools 返回 usage 对象
            last_response_content = messages[-1].get("content", [])

            # ── 没有工具调用：模型给出答案 ──────────────
            if not tool_calls:
                thought_text = ""
                final_text = ""
                for block in last_response_content:
                    if hasattr(block, "text"):
                        final_text += block.text
                        thought_text = block.text.strip()
                        break
                    elif isinstance(block, dict) and block.get("type") == "text":
                        final_text += block.get("text", "")
                        thought_text = block.get("text", "").strip()
                        break

                logger.info(f"Agent finished in {step_num} steps")
                self._print_step(
                    step_num, tool_calls, tool_results, final_text, thought_text
                )

                return AgentResult(
                    success=True,
                    answer=thought_text,
                    steps=steps,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    stop_reason="finished",
                )

            # ── 有工具调用：执行工具，追加结果 ──────────────────
            tool_results = []
            for tc in tool_calls:
                logger.info(f"  Executing: {tc.name}({tc.input})")
                current_step.tool_calls.append({"name": tc.name, "input": tc.input})
                result = self.registry.execute(tc.name, tc.input)
                result.tool_use_id = tc.id

                current_step.tool_results.append(
                    {
                        "tool": tc.name,
                        "content": result.content[:200],
                        "is_error": result.is_error,
                    }
                )
                tool_results.append(result)

            # 打印本步骤情况
            self._print_step(step_num, tool_calls, tool_results)

            # 把工具结果追加到 messages，进入下一轮
            messages = self.client.append_tool_results(messages, tool_results)
            steps.append(current_step)

            # 循环检测
            if self._detect_loop(steps):
                return AgentResult(
                    success=False,
                    answer="任务未完成：Agent 陷入循环，重复执行相同操作。",
                    steps=steps,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    stop_reason="loop_detected",
                )

        # ── 超出最大步数 ──────────────────────────────────────
        logger.warning(f"Max steps ({self.max_steps}) reached without finishing")
        return AgentResult(
            success=False,
            answer=f"任务未完成：已达到最大步数限制（{self.max_steps} 步）。",
            steps=steps,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            stop_reason="max_steps",
        )

    def _print_step(
        self, step_num, tool_calls, tool_results, final_answer="", thought=""
    ):
        """打印每步的可读日志，让 Agent 的推理过程透明可见"""
        if final_answer:
            print(f"\n✅ 最终回答（{step_num} 步完成）：")
            print(f"{final_answer}")
            return

        print(f"\n── Step {step_num} ──")

        # 打印模型的推理过程
        if thought:
            print(f"  💭 Thought: {thought[:200]}")

        for tc in tool_calls:
            print(f"  🔧 Action: {tc.name}")
            print(f"     Input:  {tc.input}")

        for res in tool_results:
            status = "❌" if res.is_error else "✓"
            preview = res.content[:150]
            print(f"  {status} Observation: {preview}")

    def _detect_loop(self, steps: list[AgentStep], window: int = 3) -> bool:
        """
        检测最近 window 步是否在重复相同的工具调用。
        如果最近三步调用的工具名+参数完全相同，判定为陷入循环。
        """
        if len(steps) < window:
            return False

        recent = steps[-window:]

        # 提取每步调用的工具签名
        def step_signature(step: AgentStep) -> str:
            return str(sorted((tc["name"], str(tc["input"])) for tc in step.tool_calls))

        signature = [step_signature(s) for s in recent]

        # 所有签名相同 → 循环调用
        if len(set(signature)) == 1 and signature[0] != "[]":
            logger.warning(
                f"Loop detected: same actions for {window} consecutive steps"
            )
            return True

        return False
