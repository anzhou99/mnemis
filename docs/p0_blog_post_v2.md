# 当我们说"AI Agent"时，我们到底在说什么？

> **Mnemis 系列 · 第 1 篇**
> 本系列记录一个独立开发者从零构建 AI Agent 产品的完整过程——学习、踩坑、架构设计、产品化。这是第一篇，从最底层的认知开始。

---

## 写在前面

网上关于 AI Agent 的文章越来越多，但大多数要么太过浅显（「Agent 就是能自动完成任务的 AI！」），要么直接跳进框架教程，让你一边复制代码一边不知道自己在做什么。

这篇文章想做一件不同的事：**从 LLM 的底层原理出发，一步步搞清楚 Agent 到底是什么，它的能力从哪里来，然后再动手把这些认知变成可以运行的代码。**

读完这篇，你会清楚地知道：
- LLM 本质上是什么，它能做什么，不能做什么
- Agent 在 LLM 的基础上扩展了哪些能力，这些能力是怎么实现的
- Temperature、Top-P、Top-K 这些参数在底层做了什么
- 如何搭建一个干净的 LLM 工程基础，作为后续开发的地基

---

## 一、LLM 是什么：一个极其强大，但也极其受限的函数

理解 Agent 之前，必须先把 LLM 的本质搞清楚。很多人用了很久的 ChatGPT，但对 LLM 的理解仍然停留在「一个很聪明的聊天机器人」这个层面。这个认知在你只是用它聊天的时候够用，但一旦你想开发 Agent，它就会成为你最大的障碍。

**LLM 的本质，是一个函数：**

```
f(输入文本) → 下一个最可能的 token
```

就这么简单，也就这么残酷。它不是一个有意识的存在，不是一个会思考的大脑，它是一个在海量文本上训练出来的、极其复杂的**概率预测机器**。给它一段文字，它预测接下来最可能出现的词（token），然后把这个词加到文字末尾，再预测下一个，如此循环，直到生成一个完整的回答。

这就是所谓的**自回归生成**（Autoregressive Generation）。

### LLM 固有的能力

因为在人类书写的几乎所有文本上训练过，LLM 获得了一些令人惊叹的**固有能力**——这些能力不需要任何额外的工程，是模型本身就有的：

| 能力 | 说明 |
|---|---|
| **语言理解与生成** | 理解问题意图，用流畅自然的语言回答 |
| **知识储备** | 训练截止日期前的大量事实性知识 |
| **推理与逻辑** | 多步骤逻辑推导，数学证明，代码理解 |
| **上下文理解** | 在给定的对话历史中理解前后语义关系 |
| **风格迁移** | 用不同的语气、风格、格式改写内容 |
| **多语言处理** | 跨语言理解与翻译 |

但 LLM 也有非常明确的**固有局限**：

| 局限 | 本质原因 |
|---|---|
| **没有记忆** | 每次调用都是独立的，不知道上次说了什么 |
| **知识有截止日期** | 训练数据有时间限制，不知道最新发生的事 |
| **无法与外部世界交互** | 不能浏览网页、读文件、调 API、执行代码 |
| **无法主动行动** | 它只能「回答」，不能「做事」 |
| **幻觉（Hallucination）** | 没有知识时，它会编造看起来合理的内容 |

这些局限不是 bug，它们是 LLM 架构本身决定的——一个「只会预测下一个 token」的函数，天然就没有持久状态，没有访问外部世界的能力。

---

## 二、Agent 是什么：给 LLM 装上手脚和记忆

理解了 LLM 的能力边界，Agent 的定义就变得非常清晰了：

> **Agent = LLM + 扩展能力的工程系统**

Agent 通过外部工程手段，系统性地补齐了 LLM 的所有短板。下面这张图是我理解 Agent 架构最清晰的方式：

```
┌─────────────────────────────────────────────────────────┐
│                        AI Agent                         │
│                                                         │
│   ┌─────────────────┐     ┌───────────────────────┐    │
│   │   LLM 核心       │ ←→  │     扩展能力层          │    │
│   │                 │     │                       │    │
│   │ • 语言理解       │     │ • 工具调用（Tool Use）  │    │
│   │ • 推理规划       │     │   → 搜索、读文件、API  │    │
│   │ • 知识储备       │     │                       │    │
│   │ • 内容生成       │     │ • 记忆系统（Memory）   │    │
│   │                 │     │   → 短期对话历史       │    │
│   └─────────────────┘     │   → 长期向量记忆       │    │
│                           │                       │    │
│                           │ • 感知层（Perception） │    │
│                           │   → 读取文件/图片/数据 │    │
│                           │                       │    │
│                           │ • 行动层（Action）     │    │
│                           │   → 执行代码、写文件   │    │
│                           └───────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

每一项扩展能力，都是用工程手段解决了 LLM 的一个固有局限：

**「没有记忆」→ 外部维护对话历史**

把每轮对话的内容存到一个数组里，下次调用时连同新问题一起发给模型。模型「看到」完整的历史，就有了「记忆」。这不是模型的能力，是你的代码在维护这个数组。

**「不知道最新信息」→ RAG（检索增强生成）**

调用 LLM 之前，先从外部知识库（网页、文档、数据库）检索相关信息，把它塞进 prompt 里。模型回答时就有了最新的上下文。

**「无法行动」→ Tool Use（工具调用）**

在 prompt 里告诉模型「你有以下工具可以使用」，模型在需要时会输出一段结构化的「我要调用这个工具」的指令，你的代码解析这段指令，真正去执行（搜索、读文件、调 API），然后把结果返回给模型。

**「不能主动规划」→ ReAct 推理循环**

让模型在回答之前先「思考」（Reasoning），输出下一步要采取的「行动」（Acting），执行后观察结果，再继续思考……这个循环就是 Agent 自主完成复杂任务的核心机制。

**关键认知**：Agent 里所有这些扩展能力，**底层都是在操控送给 LLM 的 `messages` 内容**。你把工具执行结果塞进 messages，模型看到了；你把历史记忆注入进去，模型看到了；你把检索到的文档放进去，模型看到了。LLM 始终是那个纯函数，变的是你传进去的输入。

---

## 三、LLM 是怎样「思考」的：Token 与概率

讲完宏观架构，我们往底层走一层，理解 LLM 生成每个词时到底在做什么。这不是无聊的理论——它直接决定了你应该怎么设置调用参数。

### Token：不是词，是词的碎片

LLM 处理的基本单位不是字，也不是单词，而是 **Token**——一种介于字符和单词之间的文本片段。

```
"The quick brown fox"
→ ["The", " quick", " brown", " fox"]   # 4 个 token

"人工智能正在改变世界"
→ ["人", "工", "智", "能", "正", "在", "改", "变", "世", "界"]  # 约 10 个 token
```

不同语言的 token 密度差异显著：

| 语言 | 示例 | 字符数 | Token 数 | 字符/Token |
|------|------|--------|----------|------------|
| 英文 | "artificial intelligence" | 24 | 3 | ~8 |
| 中文 | "人工智能" | 4 | 4 | ~1 |
| 代码 | `def chat(self):` | 16 | 6 | ~2.7 |

**这意味着什么？** API 按 token 计费。同样的语义内容，中文消耗的 token 约是英文的 3-5 倍。在 Agent 系统里，system prompt 每次调用都要发送，这个差异会在长期运行中产生显著的成本差距。

### 概率分布：模型在「选词」时到底在做什么

每次生成一个 token，模型实际上是在对所有可能的词汇（通常 10 万+）计算一个**概率分布**，然后从中采样。

举个例子，生成「The cat sat on the ___」的下一个词时，模型可能得到：

```
"mat"    → 42%
"floor"  → 28%  
"roof"   → 14%
"moon"   → 9%
"xyz"    → 0.1%
...（剩余词汇分摊剩下概率）
```

你如何从这个分布里「选」出最终的词，就是 Temperature、Top-P、Top-K 要控制的事情。

---

## 四、三个关键参数：Temperature、Top-P、Top-K

这三个参数是使用 LLM API 时最重要的调参工具，很多教程只告诉你「temperature 越高越随机」，却不解释为什么。我们把机制讲透。

### Temperature：拉伸或压缩概率分布

Temperature 在数学上是一个缩放因子，用于调整 softmax 函数的输出。直觉上理解：

**Temperature = 1（默认）**：使用原始概率分布。42% 的概率选 "mat"，28% 选 "floor"……

**Temperature → 0（极低）**：概率分布被「压尖」。最高概率的词（"mat"）会无限接近 100%，其他词趋近于 0。输出几乎完全确定，每次调用同一个 prompt 得到相同结果。

**Temperature → 2（极高）**：概率分布被「拉平」。所有词的概率趋于接近，"moon" 和 "xyz" 也有了不小的机会被选中。输出高度随机，经常产生意想不到（甚至胡说八道）的内容。

```
低 temperature（0.1）         高 temperature（1.5）
mat    ████████████ 92%      mat    ███ 28%
floor  █ 6%                  floor  ███ 24%
roof   ░ 1.5%                roof   ██ 18%
moon   ░ 0.4%                moon   ██ 16%
xyz    ░ 0.1%                xyz    █ 14%
```

**实践原则**：不同任务需要不同的确定性程度：

| 场景 | 推荐 Temperature | 原因 |
|------|-----------------|------|
| 结构化输出（JSON） | 0 ~ 0.1 | 格式必须精确，不能有随机波动 |
| 代码生成 | 0.1 ~ 0.3 | 逻辑正确优先，少量变体可接受 |
| 问答、摘要 | 0.3 ~ 0.7 | 准确性和自然度兼顾 |
| Agent 任务规划 | 0.2 ~ 0.5 | 需要一定创造力，但不能太飘 |
| 头脑风暴、创意写作 | 0.7 ~ 1.2 | 想要多样化的想法 |

### Top-P（核采样）：动态截断候选词

Top-P 也叫 **Nucleus Sampling**（核采样），是另一种控制随机性的方式，但机制与 Temperature 不同。

**工作原理**：把所有词按概率从高到低排序，然后从最高概率开始累加，直到累积概率达到 `p` 值，只保留这个「核」里的词，其他词一律不考虑，再从这个缩小后的候选集里采样。

```
Top-P = 0.9

mat    42%  ← 累计 42%
floor  28%  ← 累计 70%
roof   14%  ← 累计 84%
moon    9%  ← 累计 93%  ✓ 超过 0.9，截断到这里
xyz    ...  ← 不考虑
```

**为什么需要 Top-P？** Temperature 调整的是整体分布的「陡峭程度」，但有时候即使 temperature 适中，某些情况下概率分布本身就非常分散（很多词的概率差不多），Top-P 可以直接限制候选词的数量，避免模型在不该随机的地方乱选。

**Temperature 和 Top-P 如何配合使用：**

| 目标 | Temperature | Top-P |
|------|-------------|-------|
| 最确定性 | 0 | 1（不生效） |
| 日常对话 | 0.7 | 0.9 |
| 创意写作 | 1.0 | 0.95 |
| 严格结构化 | 0.1 | 1（不生效） |

一般建议：**只调其中一个，另一个保持默认**。同时调两个容易产生难以预料的效果。

### Top-K：固定候选词数量

Top-K 是最直接粗暴的截断方式：每次生成时，只保留概率最高的 K 个词，从中采样，完全忽略其他所有词。

```
Top-K = 3
只保留：mat(42%), floor(28%), roof(14%)
无论其他词概率多少，一律排除
```

**Top-K 的问题**：它是固定数量截断，没有考虑概率分布的实际形状。有时候前 K 个词的概率已经覆盖了 99.9%（分布很尖），Top-K = 50 仍然会引入很多低概率的噪音；有时候分布很平，Top-K = 3 又太激进地排除了很多合理的选项。

**Top-P 更灵活**，因为它根据概率分布的实际形状动态调整候选词数量，所以现代的 LLM 应用更常用 Top-P 而非 Top-K。Top-K 在某些专门的场景（如限制词汇多样性）和本地部署的小模型中更常见。

**Claude API 的情况**：Claude 不直接暴露 Top-K 参数，主要使用 Temperature 和 Top-P。OpenAI 的 API 也类似。如果你使用 Ollama 或 vLLM 跑本地模型，通常可以看到 Top-K 的配置项。

---

## 五、上下文窗口：模型能「看到」多远

有了 Token 的概念，上下文窗口（Context Window）就很好理解了：**每次调用 LLM 时，你能送进去的最大 token 数量**。

上下文窗口里装的是什么？

```
┌──────────────────────────────────────────┐
│              上下文窗口（200K tokens）     │
│                                          │
│  System Prompt  │  对话历史  │  新输出   │
│  （每次都发送）   │（随轮数增长）│（正在生成）│
│                                          │
└──────────────────────────────────────────┘
```

这里有几个重要推论：

**System Prompt 是固定开销**。每次 API 调用都要把 system prompt 发送一遍。如果你的 system prompt 有 2000 个中文 token，跑 1000 轮对话，光 system prompt 就消耗了 200 万 tokens。这是 Agent 开发里精简 system prompt 的动机。

**对话历史线性增长**。每轮对话结束，你要把 user 和 assistant 的内容都追加到历史里，下次调用时整个历史都要发送。对话越长，每次调用的成本越高，总有一天会超出上下文窗口上限。这就是「长对话管理」问题的根源，也是后面构建长期记忆系统的核心动机。

---

## 六、System Prompt：写给模型的行为说明书

System Prompt 不是「礼貌地介绍一下 AI 的角色」，它是你和模型之间的**行为合同**。写得好，模型会严格按你说的执行；写得含糊，模型就按自己理解的最可能方式行动。

**一个差的 System Prompt：**
```
你是一个有帮助的 AI 助手，请尽量回答用户的问题。
```

这几乎什么都没说。「有帮助」是什么？「尽量」是什么？

**一个好的 System Prompt（以 Mnemis 为例）：**
```
你是 Mnemis，一个专注于深度研究的 AI 助手。

## 行为规则
- 回答必须基于事实，不确定的内容必须明确说明「我不确定」
- 每次回答结束时，主动提出 1-2 个相关的延伸问题
- 如果问题模糊，先用一句话澄清你的理解，再回答

## 输出格式
- 优先使用 Markdown 格式
- 代码必须标注语言类型
- 长回答需要有层次分明的标题结构

## 边界
- 只处理与研究、学习、技术相关的问题
- 不生成任何误导性或有害内容
```

好的 System Prompt 有四个要素：**身份定义**（它是谁）、**行为规则**（它应该怎么做）、**输出格式**（回答的结构要求）、**边界限制**（它不应该做什么）。

你会发现，精心设计的 System Prompt 本质上是一种**对 LLM 行为的编程**——你在用自然语言写「程序」，指定这个模型实例的行为方式。这个视角会在你后面设计多 Agent 系统时变得非常重要。

---

## 七、把认知变成代码：构建 LLM 工程基础库

有了上面这些认知，我们开始动手。目标是搭一个干净的 LLM 调用基础库，作为整个 Mnemis 项目的地基。

### 项目结构

```
mnemis/
├── core/
│   ├── __init__.py
│   ├── client.py      # LLM 调用的唯一入口
│   ├── config.py      # 配置管理
│   └── models.py      # 数据结构
├── utils/
│   ├── __init__.py
│   └── logger.py      # 统一日志
├── tests/
│   └── test_client.py
├── .env               # API Key（不提交 Git）
├── .env.example       # 模板（提交 Git）
├── .gitignore
└── pyproject.toml
```

这个结构的核心设计思路：**`core/client.py` 是整个项目与 LLM 交互的唯一入口**。之后无论加 Tool Use、RAG、Multi-Agent，调用 LLM 的代码永远只在这一个地方。其他模块不直接调用 API，只通过这个 client。这叫单一职责，是让代码可维护的基本保证。

用 `uv` 初始化项目（比 pip 快 10-100 倍的现代 Python 包管理工具）：

```bash
uv init mnemis && cd mnemis
uv add anthropic pydantic pydantic-settings python-dotenv
```

### 配置管理

```python
# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str
    default_model: str = "claude-sonnet-4-5"
    max_tokens: int = 8096
    temperature: float = 0.7

settings = Settings()
```

用 Pydantic 管理配置有一个重要好处：**类型安全的启动时校验**。如果 `.env` 里漏写了 `ANTHROPIC_API_KEY`，项目启动时就会立刻报错，而不是在第一次真正调用 API 时才发现——那个时候你可能已经在一个复杂的 Agent 循环中间，很难定位问题。

### 数据模型

```python
# core/models.py
from pydantic import BaseModel
from typing import Literal

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class LLMResponse(BaseModel):
    content: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_estimate_usd(self) -> float:
        # claude-sonnet-4-5 定价：input $3/1M tokens, output $15/1M tokens
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000
```

`cost_estimate_usd` 这个属性值得单独说一下。在 Agent 开发里，一个陷入死循环或规划错误的 Agent，可以在几分钟内发起几百次 API 调用，消耗掉你的月度预算。让每次调用的成本都可见，是基本的安全意识。

### LLM 客户端

```python
# core/client.py
import anthropic
from .config import settings
from .models import Message, LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMClient:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else settings.temperature

        logger.debug(f"LLM call | messages={len(messages)} | temp={temp} | stream={stream}")

        kwargs = {
            "model": settings.default_model,
            "max_tokens": settings.max_tokens,
            "temperature": temp,
            "messages": [m.model_dump() for m in messages],
        }
        if system:
            kwargs["system"] = system

        if stream:
            return self._stream_chat(kwargs)

        response = self._client.messages.create(**kwargs)
        result = LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )
        logger.debug(
            f"Done | {result.input_tokens}in/{result.output_tokens}out"
            f" | cost≈${result.cost_estimate_usd:.5f}"
        )
        return result

    def _stream_chat(self, kwargs: dict) -> LLMResponse:
        full_text = ""
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_text += text
            print()  # 换行
            final = stream.get_final_message()
        return LLMResponse(
            content=full_text,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            model=final.model,
        )
```

---

## 八、多轮对话：「记忆」背后的真相

有了基础库，我们来实现多轮对话。这个实现过程本身就是一个很好的教学——你会亲手体验「模型的记忆完全是外部维护的」这件事。

```python
# chat_cli.py
from core.client import LLMClient
from core.models import Message

SYSTEM_PROMPT = """你是 Mnemis，一个专注于深度研究的 AI 助手。
风格：简洁、精准、有洞察力。
问题模糊时，先一句话澄清理解，再回答。"""

def run():
    client = LLMClient()
    history: list[Message] = []
    session_tokens = 0

    print("Mnemis CLI — /clear 清空历史, /tokens 查看消耗, /quit 退出\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/clear":
            history.clear()
            print("✓ 历史已清空\n")
            continue
        if user_input == "/tokens":
            print(f"本次会话消耗：{session_tokens} tokens\n")
            continue

        history.append(Message(role="user", content=user_input))
        print("Mnemis: ", end="")
        response = client.chat(messages=history, system=SYSTEM_PROMPT, stream=True)
        history.append(Message(role="assistant", content=response.content))
        session_tokens += response.total_tokens

if __name__ == "__main__":
    run()
```

运行起来后，做这个实验：

```
你: 我叫张伟
Mnemis: 你好张伟！有什么我可以帮你的？

你: 我叫什么名字？
Mnemis: 你叫张伟。

你: /clear

你: 我叫什么名字？
Mnemis: 您好！我不知道您的名字，能告诉我吗？
```

这个实验的价值不在于「验证了功能」，而在于让你真实感受到：**清空 `history` 这个 Python 列表，模型就立刻「失忆」了**。它没有任何神秘的「内置记忆」，它看到的永远只是你传进去的那个 messages 数组。

用 `/tokens` 看一下每次对话后的消耗，然后多聊几轮再看——你会感受到 token 随着历史的增长呈线性递增。这就是为什么长期运行的 Agent 需要专门的记忆管理系统。

---

## 九、结构化输出：让 LLM 成为可编程的组件

最后一个实操，也是最接近 Agent 核心机制的一步：**让 LLM 输出程序可以直接解析的结构化数据**。

为什么这很重要？回想一下 Agent 的架构：LLM 作为决策引擎，需要告诉你的程序「下一步做什么」。如果它用自然语言说「我建议你先搜索一下，然后整理结果」，你的程序没法直接执行；但如果它输出：

```json
{
  "thought": "需要先收集资料再整理",
  "action": "search",
  "action_input": "Python 异步编程最佳实践 2024",
  "is_final": false
}
```

你的程序就可以直接解析 `action` 字段，调用对应的搜索函数，把结果返回给模型继续规划。**这就是 Tool Use 的底层逻辑。**

```python
# core/client.py 中增加 structured_chat 方法
import json
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def structured_chat(
    self,
    messages: list[Message],
    response_model: Type[T],
    system: str | None = None,
    max_retries: int = 3,
) -> T:
    schema_prompt = self._build_schema_prompt(response_model)
    full_system = f"{system}\n\n{schema_prompt}" if system else schema_prompt

    last_error = None
    for attempt in range(1, max_retries + 1):
        retry_messages = messages.copy()
        if last_error and attempt > 1:
            retry_messages.append(Message(
                role="user",
                content=f"上次输出解析失败：{last_error}。请重新输出合法 JSON，不要有任何其他文字。"
            ))

        # 结构化输出用低 temperature——格式必须精确，不能随机
        response = self.chat(messages=retry_messages, system=full_system, temperature=0.1)

        try:
            raw = response.content.strip()
            # 清理 LLM 可能附加的 markdown 代码块包裹
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            return response_model(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt} failed: {e}")

    raise ValueError(f"结构化输出在 {max_retries} 次尝试后失败。最后错误：{last_error}")

@staticmethod
def _build_schema_prompt(model: Type[BaseModel]) -> str:
    fields_desc = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        desc = field.description or ""
        type_name = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
        fields_desc.append(f'  "{name}": {type_name}  // {desc}')
    return f"""只输出以下格式的 JSON，不要有任何其他内容：
{{
{chr(10).join(fields_desc)}
}}"""
```

用一个 Agent 决策场景来测试它：

```python
from pydantic import BaseModel, Field
from typing import Literal

class AgentAction(BaseModel):
    thought: str = Field(description="你的推理过程")
    action: Literal["search", "read_file", "write_file", "ask_user", "finish"] = Field(
        description="要执行的下一步动作"
    )
    action_input: str = Field(description="动作的输入参数")
    is_final: bool = Field(description="任务是否已完成")

client = LLMClient()
result = client.structured_chat(
    messages=[Message(
        role="user",
        content="任务：研究 Python 异步编程最佳实践并整理成笔记。你的下一步？"
    )],
    response_model=AgentAction,
    system="你是一个 AI Agent，每次只输出下一步行动。",
)

print(f"思考：{result.thought}")
print(f"动作：{result.action}({result.action_input})")
```

这段代码运行起来，你会得到一个可以被程序直接使用的决策结果。在 P1 阶段，我们会在这个基础上加上真正的工具执行能力，让 Agent 的决策变成真实的行动，形成完整的 ReAct 循环。

---

## 十、踩坑备忘

**Claude 的 system prompt 不在 messages 数组里**

从 OpenAI 迁移过来的经典坑。OpenAI 把 system 作为 messages 里的一条记录，Claude 是单独的 `system` 参数。把 system message 放进 Claude 的 messages 数组里，会报 `invalid role` 错误。

**API Key 绝对不能提交到 Git**

GitHub 上有专门的机器人扫描公开仓库里的 API Key，上传后可能几分钟内就被盗用，直接产生账单。正确做法：先创建 `.gitignore` 写入 `.env`，再创建 `.env` 文件，然后才 `git add`。

**Streaming 输出后要手动换行**

`print(text, end="", flush=True)` 逐 token 打印，流结束后光标停在行尾，下一行输出会接在同一行。流结束后加一个 `print()` 就解决了，不加的话日志会乱成一团。

**结构化输出时 LLM 会加 markdown 包裹**

即使你在 prompt 里说「只输出 JSON」，LLM 有时还是会输出 ` ```json ... ``` `。用 `.removeprefix("```json").removesuffix("```")` 清理一下就好。

**重试时 temperature=0.1 而非 0**

`temperature=0` 是完全确定性输出。如果第一次输出的 JSON 格式有问题，重试时会得到完全相同的结果——重试毫无意义。`0.1` 保留极微小的随机性，让重试有机会走一条略微不同的路径。

---

## 总结

这篇文章走了一条从认知到实践的路：从 LLM 的本质（一个概率预测函数）→ Agent 如何在 LLM 上扩展能力 → 理解控制输出的核心参数 → 上下文窗口与 token 的工程含义 → 动手实现基础库与多轮对话 → 结构化输出打通 Agent 决策链路。

这些认知会贯穿整个系列。每一个后续阶段的技术——Tool Use、RAG、长期记忆、Multi-Agent——本质上都是在用不同的工程手段，解决 LLM 的一个固有局限，同时通过操控送给 LLM 的 messages 内容来实现扩展能力。

理解了这个底层逻辑，你看任何 Agent 框架都会更清晰：它帮你做了什么，抽象掉了什么，以及在什么情况下你需要绕过它直接操控底层。

下一篇，我们开始给 Agent 装上「手」：**Tool Use 与 ReAct Agent**。

---

*Mnemis 系列第 1 篇 · 配套代码：github.com/your-username/mnemis*
