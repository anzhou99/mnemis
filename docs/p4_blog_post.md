# 让 AI 以团队方式工作：从零实现 Multi-Agent 协作系统

> **Mnemis 系列 · 第 5 篇**
> 前四篇我们给 Agent 装上了手脚、知识库、长期记忆。这一篇解决最后一个效率瓶颈：当任务足够复杂，单个 Agent 无论多强都力不从心——它需要一个团队。

---

## 写在前面：单 Agent 的天花板

P1 到 P3 建立的 ReAct Agent 能完成很多任务，但遇到这样的请求时开始力不从心：

> 「研究 Python 异步编程的核心机制、常见模式和性能陷阱，写一篇 2000 字的深度技术文章，包含可运行的代码示例，最后审核一遍确保技术准确。」

用单 Agent 处理这个任务：需要 10-15 步 ReAct 循环，到第 12 步时，第 2 步搜索的细节已经被后续内容稀释，模型难以同时保持「我在收集资料」和「我在构建文章结构」两种状态。而且研究、写作、审核三件事共用同一个 system prompt，每件事都做得平庸。

这是三个根本性的瓶颈，不是调参能解决的：

**注意力稀释**：LLM 的注意力机制对远距离上下文的感知是递减的。任务越复杂，步骤越多，早期信息的影响力越低。

**串行低效**：「搜索 asyncio 基础」和「搜索 asyncio 性能优化」完全独立，没有理由不能同时进行，但单 Agent 只能排队。

**专业度不足**：一个同时负责「深度搜索」「技术写作」「质量审核」的 Agent，system prompt 越来越臃肿，每件事都做得一般。

Multi-Agent 系统的出发点就是解决这三个问题：拆分职责、专门化、并行。

---

## 一、两种核心架构模式

在写第一行代码之前，需要想清楚 Agent 之间的协作关系。Multi-Agent 有两种核心模式：

**Orchestrator-Worker（中央调度）**：一个 Orchestrator 负责任务拆解、分配和聚合，Worker（Sub-agent）专注执行被分配的子任务，把结果返回给 Orchestrator。全局状态由 Orchestrator 掌控，调试容易，适合有明确分工的结构化任务。

**Peer-to-Peer（去中心化）**：Agent 之间直接互相委托任务，没有中央节点，扩展性强，但全局状态难以追踪，调试复杂。适合开放式探索任务。

对于「研究→写作→审核」这种有明确流水线结构的任务，Orchestrator-Worker 是显然的选择。

---

## 二、专门化：窄职责比宽职责好

专门化是 Multi-Agent 系统最重要的设计决策。每个 Sub-agent 的职责越窄，它能做得越好。

为什么？三个原因：

**System Prompt 更聚焦**。研究员的 system prompt 只讲「如何做好信息收集」，不需要同时讲写作规范和审核标准。模型在一件事上能调用更多「注意力」。

**工具集更精简**。只给研究员搜索工具，只给写作者文件工具，只给审核者读取权限。工具越少，模型选错工具的概率越低——P1·3 里工具子集的设计在这里发挥了价值。

**失败边界更清晰**。哪个 Agent 出问题，日志里一目了然。单 Agent 系统里，某个 step 失败往往埋在深处，难以定位。

我最终的职责划分：

```
ResearcherAgent（研究员）
  职责：信息收集，不负责写作
  工具：web_search + search_knowledge_base
  核心能力：搜索、评估信息质量、结构化输出研究笔记

WriterAgent（写作者）
  职责：内容创作，不负责研究
  工具：write_file + read_file
  核心能力：内容组织、Markdown 格式、代码示例

ReviewerAgent（审核者）
  职责：质量把控，不负责修改
  工具：read_file（只读！）
  核心能力：技术准确性、逻辑完整性、可读性评估

ResearchOrchestrator（协调者）
  职责：任务拆解、分配、监控、聚合
  工具：无（自己不执行任何操作）
  核心能力：把需求翻译成任务图，监控状态，合并输出
```

**Orchestrator 不使用任何工具**——这是有意为之。Orchestrator 只负责「想清楚做什么」，所有执行都交给 Sub-agent。这个分离让 Orchestrator 逻辑更清晰，也更容易对其行为做单元测试。

---

## 三、Agent 间通信：结构化上下文 vs 裸字符串

这是 Multi-Agent 开发里最容易被低估的细节。

初学者通常这样传递结果：

```python
# ❌ 裸字符串传递：脆弱，容易误解
researcher_output = "Python asyncio 引入了事件循环机制，gather 可以并发执行多个协程..."
writer.run(f"根据以下研究结果写文章：\n{researcher_output}")
```

问题在于：Writer 不知道哪些是确定的事实、哪些是推测；不知道来源是什么，无法生成引用；不知道哪些是重点。更严重的是，如果研究员输出了 5000 字原始搜索结果，Writer 的 context window 瞬间撑满。

结构化上下文传递要好得多：

```python
# ✅ 结构化上下文：明确、可验证
context.set_result("r1", research_notes, agent_role="researcher")
# Writer 通过 SharedContext 读取，知道这是研究员的输出
# 只取关键发现，不传递原始搜索全文
```

**LLM 处理结构化数据比处理非结构化自然语言更可靠**。当 Writer 收到的是「有来源标注的要点列表」而不是「一大段文字」，它的工作从「理解并转译」变成了「按模板填写」，输出稳定性显著提升。

我设计了 `SharedContext` 作为 Agent 间的共享状态容器——线程安全，支持按任务 ID 存取结果，同时记录通信日志便于调试：

```python
class SharedContext:
    def __init__(self, topic: str):
        self.topic = topic
        self._lock = threading.Lock()       # 线程安全，为并行执行准备
        self._task_results: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}

    def set_result(self, task_id: str, result: str, agent_role: str = ""):
        with self._lock:
            self._task_results[task_id] = result

    def get_result(self, task_id: str) -> str | None:
        with self._lock:
            return self._task_results.get(task_id)
```

---

## 四、任务状态机：五个状态，一个不能少

每个任务在执行过程中经历五种状态，每种状态都有明确含义：

```
PENDING  → 等待执行（依赖未满足，或还没轮到）
RUNNING  → 正在执行（Sub-agent 已开始工作）
DONE     → 成功完成（结果已写入 SharedContext）
FAILED   → 执行失败（异常或超时）
SKIPPED  → 被跳过（依赖任务失败，无法执行）
```

**SKIPPED 状态是关键**。当研究员失败时，写作者不能盲目开始——它没有研究结果可用。`TaskQueue` 会递归地级联跳过所有下游任务：

```python
def mark_failed(self, task_id: str, error: str):
    task.status = TaskStatus.FAILED
    self._cascade_skip(task_id)  # 递归传播

def _cascade_skip(self, failed_id: str):
    for task in self.plan.tasks:
        if failed_id in task.depends_on and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.SKIPPED
            self._cascade_skip(task.id)  # 继续向下传播
```

这个级联机制保证了系统在局部失败时能快速收敛，而不是让 Sub-agent 在没有必要上下文的情况下盲目执行产出垃圾结果。

---

## 五、并行执行：让独立任务同时运行

P4·3 实现的研究团队是串行的。但「研究 asyncio 基础」和「研究 asyncio 性能优化」完全独立，没有任何理由不能同时进行。

并行的核心是 `asyncio.gather`：

```python
# 并行执行所有 ready 任务（依赖已满足的任务）
await asyncio.gather(*[
    self._execute_single(task, queue, context)
    for task in ready_tasks
])
```

但有一个陷阱：`anthropic` Python SDK 是同步阻塞的。如果直接在协程里调用 `client.messages.create()`，它会阻塞整个事件循环，其他协程无法运行，「并行」效果完全失效。

解决方案是 `run_in_executor`，把阻塞调用放到线程池：

```python
async def run_async(self, task: AgentTask, context: SharedContext) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,        # 使用默认线程池
        self.run,    # 同步方法
        task,
        context,
    )
```

这是 asyncio 与同步库协作的标准模式：协程负责调度，线程池负责执行阻塞操作，两者通过 `run_in_executor` 桥接。

**并行的实际收益**：假设主题可以分解为两个研究子任务，各需 12 秒。串行是 24 秒，并行是 12 秒（等较慢的那个）。研究子任务越多，并行收益越显著。

---

## 六、容错的五道防线

Multi-Agent 系统失控的方式比单 Agent 多得多——每个 Agent 都可能失败，失败会以不同方式传播。我设计了从微观到宏观的五道防线：

**防线 1：Sub-agent 级超时**。每个任务最多执行 120 秒，超时后标记失败，不阻塞其他并行任务：

```python
result = await asyncio.wait_for(
    agent.run_async(task, context),
    timeout=TASK_TIMEOUT_SECONDS,
)
```

**防线 2：错误隔离**。Sub-agent 内部异常被 try/except 捕获，转换为 `TaskStatus.FAILED`，不向上传播到 `asyncio.gather` 层。并行场景下，一个任务的异常不会取消其他正在运行的任务。

**防线 3：级联跳过**。如上所述，失败任务的下游自动跳过，防止下游 Agent 在没有必要输入的情况下运行。

**防线 4：降级聚合**。`ResultAggregator` 针对各种部分成功的组合都有降级路径：有写作结果就用，没有就输出研究笔记；没有审核意见就直接输出草稿。

```python
def aggregate(self, plan, context) -> str:
    writer_results = context.get_results_by_role("writer", plan.tasks)
    if not writer_results:
        # 写作失败，降级输出研究笔记
        researcher_results = context.get_results_by_role("researcher", plan.tasks)
        return self._format_partial_output(plan.topic, researcher_results)
    # 有草稿，尝试应用审核意见
    reviewer_results = context.get_results_by_role("reviewer", plan.tasks)
    if not reviewer_results:
        return draft  # 没有审核，直接返回草稿
    return self._apply_review(draft, reviewer_results[0])
```

**防线 5：全局 Token 预算**。多 Agent 并行时，token 消耗速率是单 Agent 的 N 倍。`SharedContext` 追踪全局 token 用量，超出预算时 Orchestrator 停止分配新任务。

---

## 七、MCP：把 Agent 能力变成标准化服务

MCP（Model Context Protocol）是 Anthropic 在 2024 年发布的开放协议，目标是把「AI 如何使用工具」标准化。

在 MCP 之前，工具是代码里的函数，只能被这一个程序用。有了 MCP，工具变成了独立的服务——任何支持 MCP 的 AI 工具（Claude Desktop、Cursor、你写的 Agent 代码）都能通过标准协议使用它。

MCP Server 可以暴露三种能力：

- **Tools**：模型可以调用执行的函数（有副作用）
- **Resources**：模型可以读取的数据源（只读，URI 寻址）
- **Prompts**：预定义的提示词模板（用户在 Claude Desktop 里可以选择触发）

我给 Mnemis 实现了一个 MCP Server，把知识库和记忆能力标准化地暴露出去：

```python
# mcp_server.py
app = Server("mnemis")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="search_knowledge_base", description="..."),
        Tool(name="add_to_knowledge_base", description="..."),
        Tool(name="recall_memories", description="..."),
        Tool(name="get_user_profile", description="..."),
    ]

@app.list_resources()
async def list_resources():
    return [
        Resource(uri="mnemis://kb/sources", name="知识库文档列表"),
        Resource(uri="mnemis://memory/profile", name="用户画像"),
    ]
```

配置进 Claude Desktop 之后，在任何对话里都能直接使用 Mnemis 的知识库和记忆能力——不需要打开 Mnemis 的代码。这是从「私有工具」到「可复用服务」的跨越，也是开源后让社区其他人接入的基础。

**MCP vs 自定义 Tool Use 的选型**：

| 场景 | 选择 |
|---|---|
| 工具只被自己的 Agent 用 | 自定义 Tool Use（更简单，无额外部署） |
| 需要 Claude Desktop / Cursor 使用 | MCP（这些客户端只支持 MCP） |
| 工具要给团队其他人的 AI 工具用 | MCP（标准化，无需了解内部代码） |

---

## 八、踩坑实录

**坑 1：asyncio.gather 默认行为会取消其他任务**

`asyncio.gather` 默认 `return_exceptions=False`——任意一个协程抛出未捕获的异常，其他正在运行的协程会被立即取消。

初版代码里没有在每个 `_execute_single` 里做 try/except，导致一个研究员失败后，另一个正在运行的研究员也被取消，损失了本来已经快完成的工作。

修复：在每个 `_execute_single` 内部处理所有异常，保证不向 `gather` 层泄漏。`gather` 看到的每个协程总是「正常完成」的。

**坑 2：`run_in_executor` 线程池耗尽**

默认线程池的工作线程数是 `min(32, cpu_count + 4)`。在研究任务多（比如 5 个并行研究员）且每个 LLM 调用耗时较长的情况下，线程池被占满，新任务进入排队，并行效果大打折扣。

修复：对于大型任务，显式创建有足够容量的线程池：

```python
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
result = await loop.run_in_executor(executor, self.run, task, context)
```

**坑 3：Orchestrator 的计划解析失败导致全部崩溃**

LLM 制定计划时偶尔输出格式不符合预期的 JSON（多了解释性文字，或嵌套结构不对），`json.loads` 报错，整个任务流程在最开始就崩溃。

修复：给 `_make_plan` 加上 fallback，解析失败时使用预设的单研究员计划，保证流程总能跑起来。调试优先级远低于「能跑」：

```python
try:
    plan_data = json.loads(cleaned_response)
except Exception:
    logger.warning("Plan parsing failed, using single-researcher fallback")
    plan_data = DEFAULT_SINGLE_PLAN
```

**坑 4：写作者拿到了太多原始搜索结果**

研究员把完整的搜索结果（有时超过 3000 字）写入 SharedContext，写作者的 context window 里除了这些原始数据几乎没有空间了，写出来的文章在结构上很糟糕。

修复：研究员被要求输出**结构化研究笔记**（关键发现 + 来源 + 置信度），而不是原始搜索结果全文。在 system prompt 里明确：「你的输出是供写作者使用的研究摘要，控制在 800 字以内」。

**坑 5：MCP Server 启动时初始化 Qdrant 失败**

MCP Server 在被客户端（Claude Desktop）启动时，如果 Qdrant 没有运行，连接会报错，导致 MCP Server 进程退出，客户端看到「工具不可用」。更糟的是，这个错误发生在启动时的模块导入阶段，日志很难找到。

修复：所有需要外部服务的初始化都改为懒加载（第一次调用时才初始化），并且把工具函数用 try/except 包裹，返回错误信息而不是崩溃：

```python
_kb = None
def get_kb():
    global _kb
    if _kb is None:
        from core.knowledge_base import KnowledgeBase
        _kb = KnowledgeBase()
    return _kb

@app.call_tool()
async def call_tool(name, arguments):
    try:
        # ... 工具逻辑
    except Exception as e:
        return [TextContent(type="text", text=f"工具执行失败：{e}")]
```

---

## 九、Multi-Agent 的诚实局限性

**协调开销不可忽视**。每增加一层 Agent，就增加了一次 LLM 调用（Orchestrator 制定计划、聚合结果）。对于简单任务，Multi-Agent 比单 Agent 更慢、更贵。Multi-Agent 只在任务复杂度达到一定程度后才有净收益。

**调试难度指数级上升**。单 Agent 出问题，翻日志找到那个 step。Multi-Agent 出问题，你需要追踪任务在哪个 Agent 里出错，SharedContext 里有没有中间结果，Orchestrator 的计划是否合理。建立完善的日志体系不是可选项，是必需品。

**Orchestrator 的计划质量决定上限**。如果 Orchestrator 制定的任务拆解不合理（比如研究任务之间有隐性依赖但被当成并行处理），整个系统的输出质量会很差。LLM 生成的计划需要验证，不能盲目信任。

**并行不等于更快**。当任务受限于写作阶段（依赖所有研究结果），增加并行研究员只能减少等待时间，不能减少写作时间。真正的瓶颈在哪里，需要先分析再并行化。

---

## 十、P4 阶段全部产出

经过这个阶段，Mnemis 从「单兵作战」升级为「团队协作」：

- **`core/multi_agent/models.py`**：`AgentTask / AgentMessage / ResearchPlan`，含五状态任务机
- **`core/multi_agent/context.py`**：`SharedContext`，线程安全的共享状态容器
- **`core/multi_agent/queue.py`**：`TaskQueue`，依赖感知调度 + 级联跳过
- **`core/multi_agent/agents.py`**：`BaseSubAgent` + 三个专门化 Agent + 异步执行支持
- **`core/multi_agent/orchestrator.py`**：`ResearchOrchestrator`，支持多研究员并行
- **`core/multi_agent/scheduler.py`**：`ParallelScheduler`，asyncio 并发 + 超时控制 + 错误隔离
- **`core/multi_agent/aggregator.py`**：`ResultAggregator`，含降级处理
- **`mcp_server.py`**：Mnemis MCP Server，4 Tools + 3 Resources + 1 Prompt 模板

核心认知：Multi-Agent 的价值不在于「更多 Agent」，在于「正确的职责分工」和「合理的依赖关系」。专门化和并行是手段，高质量的输出是目的。

---

## 下一站：P5 · 自主规划与自我演进

至此 Mnemis 有了手脚、知识库、记忆、团队。

但它还有一个根本限制：**所有能力都是由我们手动设计和硬编码的**。它不会主动发现自己哪里做得不好，不会根据用户的反馈调整自己的行为，不会在没有明确指令的情况下主动规划长期目标。

P5 要让 Mnemis 第一次具备「自我意识」——能评估自己的输出质量，能根据失败经历改进行为，能把一个高层目标自动分解为多步执行计划，并在执行过程中不断修正。

这是从「执行工具」到「自主 Agent」的最后一步。

---

*Mnemis 系列第 5 篇 · 配套代码：github.com/your-username/mnemis*
