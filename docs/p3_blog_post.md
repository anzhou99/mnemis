# 给 Agent 一个会成长的记忆：从零实现长期记忆系统

> **Mnemis 系列 · 第 4 篇**
> 前三篇我们搭好了 LLM 地基、给 Agent 装上了手脚、建起了知识库。这一篇解决最后一个根本问题：如何让 Agent 真正「认识」你，而不是每次见面都像第一次。

---

## 写在前面：RAG 还不够

上一篇结束时，Mnemis 已经能查询你上传的文档了。但它仍然不认识你这个人。

你告诉它「我喜欢简洁的回答」，下次开一个新的对话窗口，它完全不记得。你们聊了三个月，它对你的了解依然是零。

这不是 RAG 能解决的问题。RAG 是「你主动放进去的文档」，记忆是「Agent 从相处中自然积累的对你的了解」。两者有根本的区别：

```
RAG 知识库：你上传 Python 文档 → Agent 能回答 Python 问题
长期记忆：你说「我不喜欢长篇大论」→ Agent 以后回答都会简洁
```

前者是工具，后者是关系。这篇文章记录了我从零实现长期记忆系统的完整过程。

---

## 一、人类记忆的三种类型：设计蓝图

认知科学把人类记忆分为三种，这个分类对 Agent 记忆系统的设计有直接指导意义。

**情景记忆（Episodic Memory）**：带时间戳的事件流。「上周三我们讨论了 RAG 架构」。有时序、有情境、会淡化。在 Agent 里对应的是**对话历史记忆**——跨会话记住我们谈过什么。

**语义记忆（Semantic Memory）**：去时序的事实。「巴黎是法国首都」。没有时间上下文，稳定、通用。在 Agent 里对应的是**用户画像记忆**——记住你是谁、你喜欢什么、你在做什么。

**程序记忆（Procedural Memory）**：技能和行为模式。「如何骑自行车」。隐性的，越用越熟练。在 Agent 里对应的是 system prompt 的动态调整——这部分在 P5「自我演进」阶段深入处理。

P3 重点实现前两种：**情景记忆 + 语义记忆**。

---

## 二、架构设计：双存储，各司其职

记忆系统需要两种查询能力：

**精确查询**：「给我所有置信度大于 0.7 的用户偏好记忆」「最近 5 次会话的摘要」——这是结构化查询，适合关系型数据库。

**语义查询**：「找和当前话题『向量数据库性能』相关的用户背景记忆」——这是向量检索，适合向量数据库。

所以我用了**双存储架构**：

```
SQLite（结构化元数据）：
  sessions 表    → 会话元信息、摘要、标题
  messages 表    → 每条对话消息
  memories 表    → 语义记忆（含置信度、类型、来源）
  memory_conflicts 表 → 冲突处理日志（审计用）

Qdrant（向量索引）：
  mnemis_memory collection → 语义记忆的向量表示
```

两者通过 `vector_id` / `memory_id` 双向关联。SQLite 是「数据库主体」，Qdrant 是「语义索引」。

---

## 三、情景记忆：对话历史的渐进式压缩

最朴素的做法是把所有历史对话原文都存起来，每次新会话时全部注入。这行不通：

- 100 轮对话 × 平均 200 tokens = 20000 tokens，直接撑爆 context window
- 即使撑得住，模型对早期内容的注意力也极低，等于无效

解决方案是**渐进式压缩**：对话进行中，每积累 20 条消息就把早期内容压缩成摘要，只保留最近 6 条原文。会话结束时生成整体摘要存入数据库。

```python
class EpisodicMemory:
    COMPRESSION_THRESHOLD = 20  # 触发压缩的消息数
    KEEP_RECENT = 6             # 保留最近几条原文

    def _compress_early_messages(self):
        to_compress = self._messages_buffer[:-self.KEEP_RECENT]
        recent = self._messages_buffer[-self.KEEP_RECENT:]
        summary = self._summarize_messages(to_compress)
        self._summaries.append(summary)
        self._messages_buffer = recent
```

传给 LLM 的 messages 变成：`[摘要1, 摘要2, 最近6条原文]`

token 消耗从「随轮数线性增长」变成「基本稳定」——两个摘要约 400 tokens，最近 6 条约 1200 tokens，总计约 1600 tokens，无论对话多长都差不多。

**为什么渐进式而不是会话结束时一次性压缩？** 有三个原因：一是对话进行中就需要用（第 30 轮的历史不能让模型一直扛着）；二是防止程序崩溃时全量丢失；三是小批量压缩的摘要质量更高，LLM 在处理 20 条时比处理 100 条能保留更多细节。

---

## 四、语义记忆：用 LLM 提炼「关于你的事实」

情景记忆记住「发生了什么」，语义记忆记住「关于你的事实」。

提炼过程完全自动化——会话结束时，把对话发给 LLM，让它抽取有长期价值的用户信息：

```python
# 提炼 Prompt 的关键设计
EXTRACTION_SYSTEM = """你是用户记忆提炼专家。
记忆类型：
- preference：用户的偏好和习惯
- fact：关于用户的客观事实
- background：用户的背景和经历
- goal：用户明确表达的目标

输出规则：
- 只提炼有长期价值的信息，忽略一次性问题和闲聊
- 置信度：明确表达=1.0，隐含推断=0.7，模糊猜测=0.5
只输出 JSON 数组。"""
```

一次典型的提炼结果：

```json
[
  {"content": "用户是 Python 后端开发者，有6年经验", "memory_type": "fact", "confidence": 1.0},
  {"content": "用户正在做名为 Mnemis 的 AI Agent 项目", "memory_type": "background", "confidence": 1.0},
  {"content": "用户偏好简洁的技术回答，不喜欢冗长解释", "memory_type": "preference", "confidence": 1.0},
  {"content": "用户目标是将项目开源并商业化", "memory_type": "goal", "confidence": 1.0}
]
```

每条记忆**双写**到 SQLite（结构化元数据）和 Qdrant（向量，支持语义检索）。

**置信度的意义**：这不只是「我们有多确定」，它在整个系统里扮演多个角色：检索过滤（只取 confidence > 0.4 的记忆）、冲突解决权重（高置信度优先）、时间衰减基础（confidence 随时间降低，低于阈值时触发遗忘）。

强迫 LLM 区分「明确说的」和「推断的」，比统一给 1.0 要合理得多。

---

## 五、记忆检索：按需注入，不是全量塞进去

有了记忆，怎么用？有两种极端的错误做法：

**错误 A：什么都不注入**。记忆存在数据库里，但模型看不到，等于没有。

**错误 B：把所有记忆都塞进 system prompt**。50 条记忆 × 平均 30 字 ≈ 1500 字 ≈ 1000 tokens，而且大多数和当前对话毫不相关，反而干扰模型判断。

正确的做法是**按需检索**：

```python
def retrieve_for_context(self, current_query: str) -> dict:
    # 语义记忆：全量取（通常 < 15 条，全部有价值）
    semantic_memories = self.semantic.get_all_by_type(
        min_confidence=0.4, limit=15
    )

    # 情景摘要：按相关性筛选 Top-3
    # 用向量相似度找和当前查询最相关的历史会话
    episodic_summaries = self._retrieve_relevant_episodes(
        query=current_query, top_k=3
    )
    return {"semantic_memories": semantic_memories, "episodic_summaries": episodic_summaries}
```

语义记忆（用户偏好/事实）全量注入——数量少，每条都有价值。情景摘要按相关性筛选——历史可能很多，只要和当前话题相关的。

最终注入 system prompt 的记忆上下文控制在 600 tokens 以内，额外成本约 $0.002 / 次对话——几乎可以忽略。

除了自动注入，Agent 还有一个 `recall_memories` 工具，可以在需要时主动检索：

```python
@tool(schema=RECALL_MEMORIES_SCHEMA)
def recall_memories(query: str, memory_type: str = None) -> ToolResult:
    memories = searcher.search_by_semantic(query, top_k=8)
    return ToolResult(content=format_memories(memories))
```

当用户问「你还记得我上次说的那个项目吗」，Agent 会主动调用这个工具，而不是只靠 system prompt 里的静态内容猜测。

---

## 六、记忆冲突：最容易被忽视的核心问题

没有冲突处理的记忆系统，是个定时炸弹。

用户三月说「我喜欢详细解释」，六月说「我觉得还是简洁一点好」。系统里同时存在两条矛盾的记忆，Agent 不知道该听哪个，行为变得混乱。

我把冲突分为三种类型，处理策略各不同：

**直接矛盾**：一条说 A，另一条说非 A。处理：新记忆覆盖旧记忆，把旧记忆的 confidence 降为 0（软删除，保留审计痕迹）。

**版本更新**：旧记忆描述了过去的状态，新记忆是进化。比如「用户在研究 RAG」→「用户在研究 Memory 系统」。处理：两者都保留，旧记忆 confidence 小幅降低。

**语义重叠**：两条说的是同一件事，措辞不同。比如「用户喜欢代码示例」和「用户偏好看代码而非文字」。处理：用 LLM 合并成一条，删除重复的。

冲突检测流程：

```python
def check_and_resolve(self, new_memory, new_vector):
    # 1. 在 Qdrant 里找相似度 > 0.85 的现有记忆
    similar = self._find_similar_memories(new_vector)
    if not similar:
        return True, new_memory  # 无冲突，直接写入

    # 2. 用 LLM 判断冲突类型
    result = self._judge_conflict(new_memory, similar[0])

    # 3. 根据类型执行解决策略
    if result["conflict_type"] == "contradiction":
        return self._resolve_contradiction(new_memory, similar[0])
    elif result["conflict_type"] == "overlap":
        return self._resolve_overlap(new_memory, similar[0], result["merged_content"])
    ...
```

**为什么用软删除（confidence→0）而不是直接删除？** 三个原因：审计追踪（可以回溯用户什么时候改变了偏好）、错误恢复（LLM 误判时可以手动恢复）、渐进式遗忘（confidence=0 的记忆在查询里自然消失，等定期清理时才真正删除）。

---

## 七、置信度时间衰减：记忆会变旧

用户三个月前说「我在研究 LangChain」，这条记忆现在还有多大参考价值？可能用户早就换到了别的框架。

我给每条记忆加了时间衰减——基于指数衰减模型：

```python
# 每隔 half_life 天，置信度降低一半
decay_factor = 0.5 ** (days_elapsed / half_life)
new_confidence = memory.confidence * decay_factor
```

不同类型的记忆，衰减速度不同：

| 类型 | 半衰期 | 原因 |
|---|---|---|
| preference | 90 天 | 偏好相对稳定，但会改变 |
| fact | 180 天 | 事实类变化最慢 |
| background | 60 天 | 当前背景变化较快 |
| goal | 45 天 | 目标最容易变化 |

当 confidence 低于 0.15 时，触发遗忘——从 SQLite 和 Qdrant 双删除。

时间衰减在每次会话结束时自动运行，通常只需几十毫秒（批量更新 SQLite，无额外 LLM 调用）。

---

## 八、踩坑实录

**坑 1：提炼出了大量「无意义」记忆**

初版 Prompt 没有明确「忽略临时性内容」，结果系统把「用户问了 Qdrant 的问题」「用户提到了 Python」这类临时性对话也存成了记忆，几天后知识库里全是噪音。

修复：在 Prompt 里明确说「只提炼有长期价值的信息，忽略一次性问题和闲聊」，并加了反例（「不要提炼：用户询问了今天的天气」）。

**坑 2：冲突检测相似度阈值设太低**

初始设了 0.7，结果「用户是 Python 开发者」和「用户在做 AI 项目」（相似度约 0.72）被判为冲突，导致大量正常的相关记忆被错误处理。

修复：把阈值提高到 0.85，这个范围内的记忆才有可能真的在说同一件事。

**坑 3：向量化在冲突检测前做，浪费 API 调用**

第一版的流程是：全部向量化 → 写入 → 检测冲突。如果冲突后记忆不被写入，向量化的 API 调用就白费了。

修复：向量化和冲突检测并行做，先用向量找候选冲突，再决定是否真的写入。批量向量化仍然先做（减少 API 调用次数），但逐条决定是否写入 Qdrant。

**坑 4：情景摘要被当成「用户说的话」**

最初我把历史摘要以 user 消息的形式注入 messages 开头，模型有时候把「上次对话中用户提到了 RAG」当成「用户在当前对话里刚说了 RAG 相关的内容」，产生语义混淆。

修复：把所有记忆（语义记忆 + 历史摘要）统一注入 system prompt，清晰地和对话历史区分开。

**坑 5：会话结束时的摘要生成偶尔超时**

摘要生成是会话结束时的同步操作。对于长对话（50+ 条消息），LLM 生成摘要需要 5-10 秒，在某些场景下用户感知到了明显延迟。

修复：把会话结束时的摘要生成和语义记忆提炼改为后台任务（`threading.Thread`），用户不需要等待，后台静默完成。

---

## 九、记忆系统的诚实局限性

**无法处理「隐性偏好变化」**：用户偏好改变了，但没有明确说出来，系统不会主动发现。记忆只来自于显式表达，无法从行为模式里推断（这是 P5 自我演进要处理的）。

**跨用户污染风险**：目前的实现没有严格的用户隔离。如果多人使用同一个 Mnemis 实例，记忆会混在一起。生产化时必须加用户 ID 隔离。

**LLM 提炼的不稳定性**：相同的对话，不同次调用可能提炼出不同的记忆（temperature=0.1 能减轻但无法完全消除）。这是基于 LLM 的记忆系统的固有不确定性。

**记忆不等于理解**：系统记住了「用户喜欢简洁回答」，但模型理解这条记忆的深度有限。在极端情况下，模型可能「知道」但「不执行」。记忆注入是提示，不是强制约束。

---

## 十、P3 阶段全部产出

经过这个阶段，Mnemis 第一次真正「认识」用户：

- **`core/memory/models.py`**：四种记忆数据类型定义
- **`core/memory/database.py`**：SQLite 操作封装，四张表
- **`core/memory/episodic.py`**：情景记忆，渐进式压缩，会话摘要
- **`core/memory/semantic.py`**：语义记忆提炼，双写存储，语义检索
- **`core/memory/retriever.py`**：按需检索，相关性 + 时序综合排序
- **`core/memory/conflict.py`**：冲突检测，时间衰减，记忆遗忘，整合
- **`core/tools/memory_tools.py`**：`recall_memories` + `summarize_user_profile`
- **`core/memory/manager.py`**：完整记忆包装层，集成进 Agent 主循环

更重要的是建立了几个认知：

记忆和知识库是不同的东西——前者自动积累，后者主动上传。情景记忆和语义记忆解决不同的问题，缺一不可。冲突处理不是可选项，是记忆系统能长期运转的前提。置信度是记忆质量的量化，也是遗忘机制的基础。软删除在不确定系统里比硬删除更安全。

---

## 下一站：P4 · Multi-Agent 与 MCP 生态

至此，Mnemis 有了手脚（Tool Use）、有了知识库（RAG）、有了记忆（Long-term Memory）。

但它还是一个独立工作的单体 Agent。面对复杂的任务——「同时研究三个技术方向并整理对比报告」——单个 Agent 的效率和可靠性都有瓶颈。

P4 要给 Mnemis 建立「团队」：Orchestrator 负责拆解任务和协调，多个专门化的 Sub-agent 并行执行，通过 MCP 协议与外部服务深度集成。第一次让 AI 以团队的方式工作。

---

*Mnemis 系列第 4 篇 · 配套代码：github.com/your-username/mnemis*
