from core.client import LLMClient
from core.models import Message
from core.tools.registry import get_registry
from core.multi_agent.models import AgentTask, AgentRole
from core.multi_agent.context import SharedContext
from utils.logger import get_logger

import asyncio

logger = get_logger(__name__)




# ── 各 Agent 的专门化 System Prompt ──────────────────────────────

RESEARCHER_SYSTEM = """你是一位专业的技术研究员，专注于信息收集和分析。

## 工作原则
- 优先使用 web_search 获取最新信息，重要内容用 search_knowledge_base 补充
- 每个关键发现都要标注来源和可信度（high/medium/low）
- 区分「确定的事实」和「观点/推测」
- 不要写作，只收集和整理信息

## 输出格式
用 Markdown 输出结构化研究笔记：
- 核心发现（每条标注来源和可信度）
- 关键数据和统计
- 相关代码示例或技术细节
- 信息来源列表"""

WRITER_SYSTEM = """你是一位专业的技术写作者，专注于将研究成果转化为高质量文章。

## 工作原则
- 基于提供的研究笔记写作，不要自行搜索或添加未经核实的信息
- 内容要有深度，不是简单罗列，要有分析和洞见
- 代码示例必须可运行，要有注释
- 结构清晰：标题层次分明，段落逻辑连贯

## 输出格式
完整的 Markdown 文章，包含：
- 吸引人的标题和引言
- 正文（多个章节，层次分明）
- 代码示例（如适用）
- 总结"""


REVIEWER_SYSTEM = """你是一位严格的技术文章审核者，专注于质量把控。

## 审核维度
1. 技术准确性：事实是否正确，代码是否可运行
2. 逻辑完整性：论点是否有支撑，推理是否严密
3. 表达清晰度：是否易于目标读者理解
4. 结构合理性：组织是否清晰，重点是否突出

## 输出格式
结构化审核报告：
- 总体评分（1-10）和一句话总评
- 具体问题列表（每条说明位置和修改建议）
- 亮点（值得保留的内容）
- 优先修改项（最重要的 2-3 条）"""


class BaseSubAgent:
    """
    Sub-agent 的基类。
    封装单次 LLM 调用（含工具支持），
    Sub-agent 不使用 ReAct 循环，只做单轮工具调用，
    保持职责简单和可预测。
    """

    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.role = role
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.llm = LLMClient()

    def run(self, task: AgentTask, context: SharedContext) -> str:
        """
        执行任务。
        构建包含任务指令和上下文的 prompt，调用 LLM（含工具支持）。
        """
        logger.info(f"[{self.role.value}] Starting task {task.id}]")

        # 从 SharedContext 读取需要的上下文数据
        messgaes = self._build_messages(task, context)

        # 获取工具列表
        registry = get_registry()
        if self.allowed_tools is None:
            tools = registry.get_schemas()
        elif len(self.allowed_tools) == 0:
            tools = []
        else:
            tools = registry.get_schemas(names=self.allowed_tools)

        # 执行（支持工具调用，最多 3 轮）
        result = self._execute_with_tools(messgaes, tools)
        logger.info(f"[{self.role.value}] Task {task.id} done ({len(result)} chars)")
        return result

    def _build_messages(self, task: AgentTask, context: SharedContext) -> list[dict]:
        """构建发给 LLM 的消息列表"""
        content = task.instruction

        # 注入上游任务结果（从 context 读取依赖任务的输出）
        upstream_results = []
        for dep_id in task.depends_on:
            dep_result = context.get_result(dep_id)
            if dep_result:
                # 找到依赖任务的角色名
                dep_role = context.get(f"task_role_{dep_id}", dep_id)
                upstream_results.append(f"## 来自 {dep_role} 的输出\n\n{dep_result}")

        if upstream_results:
            content = (
                "\n\n".join(upstream_results) + f"\n\n---\n\n## 你的任务\n\n{content}"
            )

        return [{"role": "user", "content": content}]

    def _execute_with_tools(self, messages: list[dict], tools: list) -> str:
        """执行 LLM 调用，支持最多 3 轮工具调用"""
        registry = get_registry()

        for round_num in range(1, 4):  # 最多 3 轮
            msgs, tool_calls = self.llm.chat_with_tools(
                messages=messages,
                tools=tools if tools else [],
                system=self.system_prompt,
            )

            if not tool_calls:
                # 没有工具调用，提取文本输出
                final = ""
                for block in msgs[-1].get("content", []):
                    if hasattr(block, "text"):
                        final += block.text
                    elif isinstance(block, dict) and block.get("type") == "text":
                        final += block.get("text", "")
                return final

            # 执行工具调用
            logger.debug(
                f"[{self.role.value}] Round {round_num}: "
                f"tools={[tc.name for tc in tool_calls]}"
            )

            tool_results = []
            for tc in tool_calls:
                result = registry.execute(tc.name, tc.input)
                result.tool_use_id = tc.id
                tool_results.append(result)

            messages = list(msgs)
            messages = self.llm.append_tool_results(messages, tool_results)

        # 超过最大轮数，返回最后一次的文本输出
        logger.warning(f"[{self.role.value}] Max tool rounds reached")
        return self._extract_text(messages[-1])

    @staticmethod
    def _extract_text(message: dict) -> str:
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        return "".join(
            (
                block.text
                if block and hasattr(block, "text")
                else block.get("text", "")
            )  # pyright: ignore[reportAttributeAccessIssue]
            for block in content
            if hasattr(block, "text")
            or (isinstance(block, dict) and block.get("type") == "text")
        )

    
    async def run_async(self, task: AgentTask, context: SharedContext) -> str:
        """
        异步版本的任务执行。
        实际上是把同步的 run() 放进线程池执行，
        避免 LLM API 调用阻塞事件循环。
        """
        loop = asyncio.get_event_loop()
        # run_in_executor 把同步阻塞调用放到线程池，不阻塞事件循环
        result = await loop.run_in_executor(None, self.run, task, context)
        return result

# ── 三个具体的 Sub-agent ──────────────────────────────────────────


class ResearcherAgent(BaseSubAgent):
    def __init__(self):
        super().__init__(
            role=AgentRole.RESEARCHER,
            system_prompt=RESEARCHER_SYSTEM,
            allowed_tools=["web_search", "search_knowledge_base"],
        )


class WriterAgent(BaseSubAgent):
    def __init__(self):
        super().__init__(
            role=AgentRole.WRITER,
            system_prompt=WRITER_SYSTEM,
            allowed_tools=["write_file", "read_file"],
        )


class ReviewerAgent(BaseSubAgent):
    def __init__(self):
        super().__init__(
            role=AgentRole.REVIEWER,
            system_prompt=REVIEWER_SYSTEM,
            allowed_tools=["read_file"],  # 只读，不能写
        )
