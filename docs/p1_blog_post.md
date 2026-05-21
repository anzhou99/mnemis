# 给 AI 装上手脚：从零实现 ReAct Agent

> **Mnemis 系列 · 第 2 篇**
> 上一篇我们搭好了 LLM 调用的工程地基。这一篇，我们给 Agent 装上「手脚」——让它能搜索网络、读写文件，并用 ReAct 循环自主完成多步任务。全程手写，不依赖 LangChain。

---

## 写在前面：为什么不直接用 LangChain

这是我在开始 P1 阶段时问自己的第一个问题。LangChain 有现成的 Agent 实现，几行代码就能跑起来。

但我最终选择手写，原因很简单：**你不理解的代码，在出问题时会让你完全束手无策。**

Tool Use 和 ReAct 的实现并不复杂——核心逻辑不超过 200 行代码。手写一遍之后，你对每一个设计决策都有清晰的认知。之后再看 LangChain、LlamaIndex 的源码，你会发现它们做的事情和你手写的完全一样，只是加了更多抽象层。

这篇文章就是这 200 行代码背后的完整思路。

---

## 一、Tool Use 不是魔法，是一个协议

在写第一行代码之前，先把最重要的认知说清楚：

**模型没有真正「调用」工具的能力。** 它唯一能做的还是生成文本。

Tool Use 是你（开发者）、模型、工具执行环境三方之间约定的**消息协议**。完整流程分六步：

```
① 你发送：messages + tools 定义（告诉模型有哪些工具可用）
② 模型决策：需要工具吗？用哪个？参数是什么？
③ 模型返回：tool_use block（不是直接执行，是「请求调用」）
④ 你的代码：解析 tool_use，真正去执行工具函数
⑤ 你追加：tool_result 到 messages，回传给模型
⑥ 模型回答：看到工具结果后，给出最终文本回答
```

模型只负责 ②③⑥，真正执行工具（④）的永远是你的代码。

理解这一点，Tool Use 就从「魔法」变成了「协议」——一个你完全能掌控的工程系统。

### 消息格式：不要靠猜，把原始数据看一遍

第③步，模型返回的 `response.content` 里包含 `tool_use` block：

```python
response.content = [
    {
        "type": "text",
        "text": "让我搜索一下最新价格。"   # 模型的思考，可能为空
    },
    {
        "type": "tool_use",
        "id": "toolu_01XFw3...",      # ← 唯一 ID，回传时必须用这个
        "name": "web_search",
        "input": {"query": "比特币今日价格 2025"}
    }
]
# stop_reason == "tool_use" 是「模型要调工具」的信号
```

第⑤步，你把结果以 `user` 角色回传：

```python
{
    "role": "user",          # ← 注意是 user，不是 tool
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01XFw3...",  # ← 对应上面的 id
            "content": "比特币：$67,234，+2.3%"
        }
    ]
}
```

**为什么 tool_result 是 user 角色？** 从模型视角看，工具结果是「外部世界反馈的信息」，不是模型自己说的话，所以归入 user 角色。这个设计初看反直觉，但理解后就不会搞混了。

---

## 二、Tool Schema：你在教模型「何时用、怎么用」

很多人觉得 Schema 只是「把函数参数描述一遍」，这是严重低估。

**Tool Schema 本质上是你在用自然语言给模型编程**——告诉它什么情况下该调这个工具，传什么样的参数。

对比这两个 Schema：

```python
# ❌ 差的 Schema：模型不知道什么时候该用
{
    "name": "search",
    "description": "搜索信息",    # 太模糊
    "input_schema": {
        "properties": {
            "q": {"type": "string"}  # 参数名不清晰，没有说明
        }
    }
}

# ✅ 好的 Schema：模型知道何时用、怎么用
{
    "name": "web_search",
    "description": """搜索互联网获取实时信息。
    适用：今日价格、最新新闻、近期事件
    不适用：通用知识、数学、历史定论""",
    "input_schema": {
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索词，建议精确，可加时间限定如'2025年'"
            }
        },
        "required": ["query"]
    }
}
```

我在测试中发现：description 写得差的工具，模型要么在不该用的时候用，要么有需要时不用。**description 是工具调用准确率最重要的影响因素，没有之一。**

---

## 三、数据结构设计：三个类，对应三个关键时刻

Tool Use 流程里你需要处理三个关键时刻，对应三个数据结构：

```python
# core/tools/base.py

class ToolSchema(BaseModel):
    """工具定义，传给 LLM 告知有哪些工具"""
    name: str
    description: str
    input_schema: dict

class ToolCall(BaseModel):
    """解析模型返回的 tool_use block"""
    id: str      # 回传时必须对应
    name: str
    input: dict

class ToolResult(BaseModel):
    """封装执行结果，准备回传"""
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_api_format(self) -> dict:
        result = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            result["is_error"] = True
        return result
```

`is_error` 这个字段值得特别说明：**工具失败时不应该抛出异常，而应该返回 `is_error=True` 的结果，让模型自己决定如何应对。** 这是让 Agent 能自我修正的基础。

---

## 四、插件化工具注册系统

实现了第一个工具后，我意识到手动维护工具列表是个陷阱：

```python
# 手动维护：每次加新工具都要改两个地方，迟早漏掉
TOOLS = [WEB_SEARCH_SCHEMA, READ_FILE_SCHEMA, WRITE_FILE_SCHEMA]
TOOL_EXECUTORS = {
    "web_search": lambda args: execute_web_search(**args),
    "read_file": lambda args: execute_read_file(**args),
    ...
}
```

解决方案是一个 `@tool` 装饰器 + 全局注册表：

```python
# 定义工具时只需加一行装饰器
@tool(schema=WEB_SEARCH_SCHEMA)
def execute_web_search(query: str, max_results: int = 3) -> ToolResult:
    ...

# 使用时，注册表自动知道有哪些工具
registry = get_registry()
registry.get_schemas()              # 所有工具 Schema
registry.execute("web_search", {    # 执行工具
    "query": "Python 3.13 新特性"
})
```

注册表是一个全局单例，`@tool` 装饰器在模块被 import 时自动把工具注册进去。`core/tools/__init__.py` 统一 import 所有工具模块，保证注册表在使用前已经填满：

```python
# core/tools/__init__.py
from core.tools import search      # noqa: F401 — 触发 @tool 装饰器
from core.tools import file_ops   # noqa: F401
# 未来新增工具：在这里加一行就够了
```

这个设计遵循**开闭原则**：对扩展开放（新增工具只加 `@tool`），对修改关闭（不需要改 Agent 或注册表代码）。

注册表还支持**工具子集**——给不同的 Agent 配不同的工具权限：

```python
# 搜索 Agent 只能用搜索工具
search_tools = registry.get_schemas(names=["web_search"])

# 写作 Agent 只能读写文件
file_tools = registry.get_schemas(names=["read_file", "write_file"])
```

限制工具子集有两个好处：安全性（Agent 只能做它该做的事）和准确性（工具越少，模型选择越精准，减少幻觉调用）。

---

## 五、ReAct：不只是「调工具的循环」

实现工具之后，下一个问题是：如何让 Agent 自主完成**多步骤任务**？

单次 Tool Use 只能「问一次、用一次工具、给一次答案」，对于需要动态调整的复杂任务远远不够。

**ReAct（Reasoning + Acting）** 解决了这个问题，它来自 2022 年的一篇论文，核心思想是：**让模型在每次行动之前显式地推理，把思考和行动交替进行，形成循环。**

```
用户提问
  ↓
Thought：我需要先搜索 Python 3.13 的信息
  ↓
Action：web_search("Python 3.13 new features")
  ↓
Observation：[搜索结果...]
  ↓
Thought：已有足够信息，现在整理成 Markdown 写入文件
  ↓
Action：write_file("notes.md", "# Python 3.13...")
  ↓
Observation：✓ 已写入
  ↓
最终回答：我已完成研究并保存到 notes.md...
```

这个循环让 Agent 能**根据每步结果动态调整计划**，而不是在开始时就把所有步骤写死。

### messages 在循环中是怎么增长的

这是实现 ReAct 最容易搞混的地方。每轮循环后，messages 数组都会追加新内容：

```
初始：
  [user: "研究 Python 3.13 并保存到文件"]

第1轮结束后：
  [user: "...",
   assistant: [tool_use(id="abc", web_search, ...)],
   user: [tool_result(id="abc", "搜索结果...")]]

第2轮结束后：
  [...前3条...,
   assistant: [tool_use(id="xyz", write_file, ...)],
   user: [tool_result(id="xyz", "✓ 已写入")]]

第3轮（最终）：
  [...前5条...,
   assistant: [text("我已完成研究...")]]
```

模型每次看到完整的历史——包括它之前的所有行动和结果。**这就是 ReAct「自我修正」能力的来源**：它能看到上一步做了什么、结果如何，再决定下一步。

---

## 六、ReAct 核心实现：不到 100 行

```python
# core/agent.py（核心逻辑简化版）

class ReActAgent:
    def run(self, task: str) -> AgentResult:
        tools = self.registry.get_schemas(self.tool_names)
        messages = [{"role": "user", "content": task}]
        steps = []

        for step_num in range(1, self.max_steps + 1):

            # 预算检查
            if self._over_budget():
                return AgentResult(success=False, stop_reason="budget_exceeded", ...)

            # 调用 LLM
            messages, tool_calls, usage = self.client.chat_with_tools(
                messages=messages, tools=tools, system=self.system_prompt
            )
            self._update_tokens(usage)

            # 没有工具调用 → 模型给出了最终答案
            if not tool_calls:
                return AgentResult(success=True, answer=self._extract_text(messages), ...)

            # 执行所有工具，追加结果
            tool_results = []
            for tc in tool_calls:
                result = self._execute_with_retry(tc.name, tc.input)
                result.tool_use_id = tc.id
                tool_results.append(result)

            messages = self.client.append_tool_results(messages, tool_results)
            steps.append(AgentStep(...))

            # 循环检测
            if self._detect_loop(steps):
                return AgentResult(success=False, stop_reason="loop_detected", ...)

        return AgentResult(success=False, stop_reason="max_steps", ...)
```

整个 ReAct 循环的逻辑非常线性：调 LLM → 有工具就执行 → 没工具就结束。复杂性都在边界情况的处理上，这正是下一节的重点。

---

## 七、让 Agent 可信赖：五道防线

一个「能跑」的 Agent 和一个「可信赖」的 Agent 之间，差距在于它如何处理各种意外情况。

我给 ReActAgent 加了五道防线：

### 防线 1：最大步数 + 循环检测

最大步数是最基本的保险。循环检测更智能——如果最近三步调用了完全相同的工具和参数，判定为陷入循环，提前终止：

```python
def _detect_loop(self, steps, window=3) -> bool:
    if len(steps) < window:
        return False
    signatures = [str(sorted((tc["name"], str(tc["input"]))
                              for tc in s.tool_calls))
                  for s in steps[-window:]]
    return len(set(signatures)) == 1 and signatures[0] != "[]"
```

### 防线 2：工具执行超时

网络工具在网络不稳定时可能卡住，用 `ThreadPoolExecutor` 实现超时控制：

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(fn, **tool_input)
    try:
        return future.result(timeout=30.0)
    except concurrent.futures.TimeoutError:
        return ToolResult(content="执行超时，请稍后重试", is_error=True)
```

这里有个细节：为什么用线程池而不是 `asyncio`？因为我们的工具函数是同步的，`asyncio.wait_for` 只对 `async def` 有效。

### 防线 3：错误隔离 + 智能重试

关键设计：**工具错误不应该崩溃 Agent，应该以 `is_error=True` 的结果回传给模型，让模型自己决定如何应对。**

重试策略也需要区分错误类型：超时错误值得重试，参数错误不值得重试（重试结果完全一样，应该让模型修正参数）：

```python
def _execute_with_retry(self, name, tool_input, max_retries=2):
    for attempt in range(1, max_retries + 1):
        result = self.registry.execute(name, tool_input)
        if not result.is_error:
            return result
        if "参数错误" in result.content:
            return result   # 不重试，让模型修正
        if attempt < max_retries:
            logger.warning(f"Retrying {name} (attempt {attempt})...")
    return result
```

### 防线 4：Token 预算控制

精确追踪每轮调用的 token 用量，累计超出预算时立即中止。一个陷入错误循环的 Agent 可以在几分钟内消耗掉大量 token，这道防线是成本的最后一道屏障。

### 防线 5：Human-in-the-Loop

对于不可逆的高风险操作（写文件、发邮件、删数据），Agent 执行前暂停请求用户确认：

```python
agent = ReActAgent(
    require_confirm=["write_file", "send_email"],
)
```

用户拒绝时，拒绝信息作为 `tool_result` 回传给模型，Agent 会自动调整方案继续——而不是直接崩溃。

这五道防线背后有一个共同的设计原则：**Graceful Degradation（优雅降级）**。每一种失败模式都有对应的降级策略，Agent 尽可能在内部消化问题，给用户返回有意义的结果，而不是一个裸露的异常堆栈。

---

## 八、真实运行效果

加上完整防护后，一次典型的 Agent 运行日志是这样的：

```
── Step 1 ──
  💭 Thought: 需要搜索 Python 3.13 的最新特性信息
  🔧 Action:  web_search({'query': 'Python 3.13 new features 2024'})
  ✓  Observation: Python 3.13 带来了改进的错误消息、实验性 JIT 编译器...

── Step 2 ──
  💭 Thought: 已获取足够信息，整理成 Markdown 写入文件
  🔧 Action:  write_file({'filename': 'python313.md', 'content': '...'})

⚠️  Agent 请求执行写文件操作，需要确认：
   filename: python313.md
   是否允许？(y/n): y

  ✓  Observation: ✓ 已写入 python313.md（1847 字符）

✅ 最终回答（2 步完成）：
我已完成研究，将 Python 3.13 的主要新特性整理保存到 python313.md...

📊 统计：2 步 | 1,243 tokens | $0.0089
```

Thought / Action / Observation 清晰可见，Human-in-the-loop 在关键节点介入，成本透明。

---

## 九、踩坑实录

**坑 1：把 assistant 的 content 只存文本，丢掉了 tool_use block**

工具调用结束后追加 messages 时，必须把 `response.content`（原始列表，包含 tool_use block）存进去，而不是只提取 text：

```python
# ❌ 错误：只存了文本，丢失了 tool_use block
messages.append({"role": "assistant", "content": response.content[0].text})

# ✅ 正确：存原始 content 列表
messages.append({"role": "assistant", "content": response.content})
```

这个错误会导致 Claude 报 `tool_use_id not found` 错误，因为模型找不到它之前发出的 tool_use block。

**坑 2：Parallel Tool Use 时只处理了第一个工具**

当模型一次返回多个 `tool_use` block，必须全部执行并全部回传结果。如果只处理第一个，模型会困惑为什么另一个工具没有结果：

```python
# ❌ 错误：只处理了第一个
tool_call = tool_calls[0]

# ✅ 正确：遍历所有工具调用
for tc in tool_calls:
    result = registry.execute(tc.name, tc.input)
    ...
```

**坑 3：工具的路径安全检查没做**

文件读写工具如果不做路径限制，模型可能（无论是幻觉还是 prompt 注入）传入 `../../etc/passwd` 这样的危险路径。用 `Path(filename).name` 只取文件名部分，强制限制在 workspace 目录里：

```python
safe_path = WORKSPACE_DIR / Path(filename).name  # 只取文件名，去掉路径
```

**坑 4：循环检测的时机错了**

最初我在每轮开始时检测，但这样会漏掉第一次循环。正确做法是在工具执行完、追加结果之后检测，因为 `steps` 里已经有了本轮的行动记录：

```python
steps.append(current_step)
if self._detect_loop(steps):  # 追加后再检测
    return ...
```

**坑 5：温度设置对工具调用有影响**

在结构化输出阶段我就知道低温度有助于格式稳定。在 ReAct 里也一样：Temperature 过高时，模型有时会在不需要工具的情况下随机调用工具，或者用不合适的参数。Agent 的 LLM 调用建议用 `temperature=0.3` 左右，兼顾准确性和一定的灵活性。

---

## 十、总结与下一步

P1 阶段我们做了四件事：

**理解了 Tool Use 的本质**：不是魔法，是你和模型之间的消息协议。模型只「决定」调用什么工具，真正执行的是你的代码。

**建立了可扩展的工具系统**：`@tool` 装饰器 + 全局注册表，新增工具一行代码，不改任何其他地方。

**实现了 ReAct 循环**：Thought → Action → Observation 的循环让 Agent 能动态调整计划，自主完成多步任务。

**让 Agent 可信赖**：五道防线（循环检测、工具超时、错误隔离、预算控制、Human-in-the-loop）让 Agent 从「实验室能跑」升级到「真实环境可用」。

至此，Mnemis 的 Agent 已经有了手脚。它能搜索信息、读写文件、自主规划多步任务——这已经是一个有实际价值的工具了。

但还缺少一个关键能力：**知识**。目前 Agent 依赖实时搜索来获取信息，无法利用你已有的文档、笔记、资料。

下一篇，我们构建 RAG 知识库系统——让 Agent 拥有可以持续积累的「长期知识」。

---

*Mnemis 系列第 2 篇 · 配套代码：github.com/your-username/mnemis*
