# core/eval/cases.py
from core.eval.models import EvalCase, EvalDimension

# ── Mnemis 标准测试用例集 ─────────────────────────────────────────
# 覆盖五个核心能力维度，共 20 个测试用例

MNEMIS_EVAL_CASES: list[EvalCase] = [

    # === 工具调用准确性（4个用例）===

    EvalCase(
        id="tc-001",
        name="需要实时信息时应调用搜索工具",
        input="今天 Python 3.13 的最新版本号是多少？",
        expected_behavior="应调用 web_search 工具获取实时信息，不能凭记忆回答",
        dimensions=[EvalDimension.TOOL_SELECTION, EvalDimension.FACTUAL_ACCURACY],
        tags=["tool_use", "realtime"],
    ),
    EvalCase(
        id="tc-002",
        name="不需要工具时不应调用工具",
        input="用 Python 写一个冒泡排序函数",
        expected_behavior="直接用知识回答，不需要调用任何搜索或知识库工具",
        dimensions=[EvalDimension.TOOL_SELECTION, EvalDimension.INSTRUCTION_FOLLOW],
        tags=["tool_use", "coding"],
    ),
    EvalCase(
        id="tc-003",
        name="知识库问题优先查知识库",
        input="根据我的文档，Mnemis 项目的技术架构是什么？",
        expected_behavior="应优先调用 search_knowledge_base，而非 web_search",
        dimensions=[EvalDimension.TOOL_SELECTION],
        tags=["tool_use", "rag"],
    ),
    EvalCase(
        id="tc-004",
        name="记忆相关问题应调用记忆工具",
        input="你还记得我之前告诉你我在做什么项目吗？",
        expected_behavior="应调用 recall_memories 工具查询长期记忆",
        dimensions=[EvalDimension.TOOL_SELECTION],
        tags=["tool_use", "memory"],
    ),

    # === 回答质量（4个用例）===

    EvalCase(
        id="qa-001",
        name="技术解释的深度和准确性",
        input="请解释 Python asyncio 的事件循环是如何工作的",
        expected_behavior=(
            "回答应涵盖：事件循环的基本原理、协程的调度机制、I/O 复用的概念。"
            "应有代码示例，逻辑清晰，适合有 Python 基础的开发者。"
        ),
        reference_answer=(
            "事件循环是 asyncio 的核心，负责调度协程的执行。"
            "当协程遇到 await 时，控制权交还给事件循环，"
            "事件循环可以继续执行其他就绪的协程..."
        ),
        dimensions=[EvalDimension.ANSWER_QUALITY, EvalDimension.FACTUAL_ACCURACY],
        tags=["qa", "technical"],
    ),
    EvalCase(
        id="qa-002",
        name="对比分析的完整性",
        input="比较 Qdrant 和 Chroma 这两个向量数据库的优缺点",
        expected_behavior=(
            "应从多个维度对比：性能、易用性、生产就绪程度、开源协议、适用场景。"
            "结论应有明确的选型建议。"
        ),
        dimensions=[EvalDimension.ANSWER_QUALITY, EvalDimension.INSTRUCTION_FOLLOW],
        tags=["qa", "comparison"],
    ),
    EvalCase(
        id="qa-003",
        name="代码生成的正确性和可运行性",
        input="写一个 Python 函数，使用 asyncio.gather 并发执行三个异步任务，并处理其中一个失败的情况",
        expected_behavior=(
            "代码应可直接运行，使用 asyncio.gather 的 return_exceptions=True 或独立 try/except，"
            "有异常处理，有简短注释说明关键部分。"
        ),
        dimensions=[
            EvalDimension.ANSWER_QUALITY,
            EvalDimension.FACTUAL_ACCURACY,
            EvalDimension.INSTRUCTION_FOLLOW,
        ],
        tags=["qa", "coding"],
        weight=1.5,  # 代码生成权重更高
    ),
    EvalCase(
        id="qa-004",
        name="简洁回答的适度性",
        input="Python 的 GIL 是什么，一句话解释",
        expected_behavior="回答应简洁，不超过3句话，直接点明 GIL 的核心概念",
        dimensions=[EvalDimension.INSTRUCTION_FOLLOW, EvalDimension.ANSWER_QUALITY],
        tags=["qa", "concise"],
    ),

    # === 事实准确性（4个用例）===

    EvalCase(
        id="fa-001",
        name="知识边界的诚实性",
        input="Mnemis 项目的 GitHub star 数量是多少？",
        expected_behavior=(
            "应诚实说明无法获取此信息（这是私有信息或需要实时查询），"
            "而不是编造一个数字。"
        ),
        dimensions=[EvalDimension.HALLUCINATION_FREE, EvalDimension.FACTUAL_ACCURACY],
        tags=["honesty", "hallucination"],
    ),
    EvalCase(
        id="fa-002",
        name="技术细节的准确性",
        input="Python 的 list.sort() 使用的是什么排序算法？时间复杂度是多少？",
        expected_behavior=(
            "应准确回答：Timsort 算法，时间复杂度 O(n log n) 最坏，"
            "O(n) 最好（已排序数据）。"
        ),
        reference_answer="Python 的 list.sort() 使用 Timsort 算法，最坏时间复杂度 O(n log n)，最好 O(n)。",
        dimensions=[EvalDimension.FACTUAL_ACCURACY],
        tags=["factual", "technical"],
    ),
    EvalCase(
        id="fa-003",
        name="版本信息的准确性和时效性意识",
        input="Claude API 目前支持哪些模型？",
        expected_behavior=(
            "应承认模型列表可能不是最新的，建议查看官方文档，"
            "或使用 web_search 获取最新信息。"
        ),
        dimensions=[EvalDimension.FACTUAL_ACCURACY, EvalDimension.HALLUCINATION_FREE],
        tags=["factual", "temporal"],
    ),
    EvalCase(
        id="fa-004",
        name="拒绝编造不确定信息",
        input="明天上海的天气怎么样，适合出门吗？",
        expected_behavior=(
            "应使用 web_search 搜索天气，或说明需要实时天气数据。"
            "不应直接编造天气信息。"
        ),
        dimensions=[EvalDimension.HALLUCINATION_FREE, EvalDimension.TOOL_SELECTION],
        tags=["honesty", "realtime"],
    ),

    # === 指令遵循度（4个用例）===

    EvalCase(
        id="if-001",
        name="格式要求的遵循",
        input="用 JSON 格式列出5种常见的 Python 设计模式，每个包含名称和一句话说明",
        expected_behavior="输出必须是有效的 JSON 格式，包含恰好5个设计模式，每个有 name 和 description 字段",
        dimensions=[EvalDimension.INSTRUCTION_FOLLOW],
        tags=["format", "structured_output"],
        weight=1.5,
    ),
    EvalCase(
        id="if-002",
        name="字数限制的遵循",
        input="用不超过100字解释什么是 RAG",
        expected_behavior="回答不超过100字，同时覆盖 RAG 的核心概念",
        dimensions=[EvalDimension.INSTRUCTION_FOLLOW, EvalDimension.ANSWER_QUALITY],
        tags=["format", "concise"],
    ),
    EvalCase(
        id="if-003",
        name="步骤化输出的遵循",
        input="分步骤解释如何在 Python 中实现一个简单的 LRU Cache，不使用 functools",
        expected_behavior="应给出编号的步骤，逻辑清晰，最后有完整可运行的代码",
        dimensions=[EvalDimension.INSTRUCTION_FOLLOW, EvalDimension.ANSWER_QUALITY],
        tags=["format", "coding"],
    ),
    EvalCase(
        id="if-004",
        name="角色扮演的准确性",
        input="请以一位有10年经验的 Python 架构师的视角，评价我选择用 FastAPI + async SQLAlchemy 的技术栈",
        expected_behavior=(
            "应从资深架构师视角出发，给出有深度的评价，"
            "包括优点、潜在风险和具体建议，语气专业而不说教。"
        ),
        dimensions=[EvalDimension.INSTRUCTION_FOLLOW, EvalDimension.ANSWER_QUALITY],
        tags=["roleplay", "technical"],
    ),

    # === 综合能力（4个用例）===

    EvalCase(
        id="comp-001",
        name="多步任务的完整执行",
        input="搜索 Python 3.13 的三个最重要的新特性，然后把结果保存到 python313_features.md 文件",
        expected_behavior=(
            "应先调用 web_search，然后调用 write_file 保存结果，"
            "文件内容应有清晰的结构，包含至少3个新特性的说明。"
        ),
        dimensions=[
            EvalDimension.TOOL_SELECTION,
            EvalDimension.INSTRUCTION_FOLLOW,
            EvalDimension.FACTUAL_ACCURACY,
        ],
        tags=["multistep", "tool_use"],
        weight=2.0,
    ),
    EvalCase(
        id="comp-002",
        name="信息整合和对比",
        input="对比 asyncio 和 threading 在 Python 中的适用场景，给出选择建议",
        expected_behavior=(
            "应涵盖 I/O 密集 vs CPU 密集的区别，GIL 的影响，"
            "给出清晰的选型决策树或建议，有具体示例。"
        ),
        dimensions=[EvalDimension.ANSWER_QUALITY, EvalDimension.FACTUAL_ACCURACY],
        tags=["comparison", "technical"],
    ),
    EvalCase(
        id="comp-003",
        name="处理模糊需求",
        input="帮我优化这段代码",  # 故意不给代码
        expected_behavior=(
            "应要求用户提供代码，不应编造或猜测代码内容。"
            "回应应礼貌地说明需要看到实际代码。"
        ),
        dimensions=[EvalDimension.INSTRUCTION_FOLLOW, EvalDimension.HALLUCINATION_FREE],
        tags=["edge_case", "clarification"],
    ),
    EvalCase(
        id="comp-004",
        name="长对话的连贯性",
        input="我们刚才聊的那个话题，你能总结一下吗？",  # 在没有历史上下文时
        expected_behavior=(
            "应说明当前没有之前对话的上下文，"
            "或尝试调用 recall_memories 查询历史。"
            "不应编造之前的对话内容。"
        ),
        dimensions=[EvalDimension.HALLUCINATION_FREE, EvalDimension.TOOL_SELECTION],
        tags=["edge_case", "memory"],
    ),
]

# 按标签分组，方便选择性运行
def get_cases_by_tag(tag: str) -> list[EvalCase]:
    return [c for c in MNEMIS_EVAL_CASES if tag in c.tags]

def get_cases_by_dimension(dim: EvalDimension) -> list[EvalCase]:
    return [c for c in MNEMIS_EVAL_CASES if dim in c.dimensions]