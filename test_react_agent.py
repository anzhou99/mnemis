# run_agent.py
import core.tools  # 触发所有工具注册
from core.agent import ReActAgent

# 在 run_agent.py 或 Agent 初始化时使用这个 System Prompt

RAG_ENABLED_SYSTEM = """你是 Mnemis，一个拥有私有知识库的 AI 研究助手。

## 信息源决策策略
你有以下信息来源，按优先级使用：

1. **私有知识库**（search_knowledge_base）
   - 用户提到「我的文档」「之前添加的」「知识库里」时优先使用
   - 问题涉及内部资料、个人笔记时使用
   - 先查知识库，找到相关内容再回答

2. **网络搜索**（web_search）
   - 需要实时信息（新闻、当前价格、最新版本）时使用
   - 知识库没有相关内容时，转向网络搜索

3. **自有知识**
   - 通用知识、数学、编程基础等，不需要查询任何工具

## 引用规范
- 使用知识库内容时，在回答中用 [数字] 标注来源
- 使用网络信息时，说明信息来源
- 不确定时主动说明，不要编造

## 记忆管理
- 用户要求「记住」某内容时，使用 add_to_knowledge_base 工具
- 用户问「知识库里有什么」时，使用 list_knowledge_base_sources"""


def main():
    agent = ReActAgent(
        system_prompt=RAG_ENABLED_SYSTEM,
        max_steps=6,
    )

    # ── 第一步：先往知识库里存一些内容 ──────────────────────────
    print("=" * 60)
    print("第1步：向知识库添加内容")
    print("=" * 60)

    result = agent.run(
        """请把以下内容添加到知识库，来源标识为 'mnemis_architecture'：

        Mnemis 是一个 AI Agent 项目，采用以下技术架构：
        - LLM 核心：Claude claude-sonnet-4-5，负责推理和决策
        - 工具系统：基于 @tool 装饰器的插件化注册表
        - 向量知识库：Qdrant + OpenAI Embedding
        - 检索策略：Hybrid Search（向量 + BM25 + RRF 融合）
        - Agent 循环：ReAct 模式（Thought → Action → Observation）
        - 安全机制：最大步数限制、循环检测、Token 预算控制"""
    )
    print(f"\n状态：{'✅' if result.success else '❌'} | {result.stop_reason}\n")

    # ── 第二步：查询刚添加的内容 ─────────────────────────────────
    print("=" * 60)
    print("第2步：查询知识库内容")
    print("=" * 60)

    result = agent.run("Mnemis 的检索策略是什么？它用了哪些技术来提升检索质量？")
    print(f"\n状态：{'✅' if result.success else '❌'} | {result.stop_reason}\n")

    # ── 第三步：知识库 + 网络搜索协作 ────────────────────────────
    print("=" * 60)
    print("第3步：知识库 + 网络搜索协作")
    print("=" * 60)

    result = agent.run(
        "根据知识库里的 Mnemis 架构信息，再结合网络上最新的 Claude API 资料，"
        "告诉我 Mnemis 当前使用的模型有哪些值得关注的新特性？"
    )
    print(f"\n状态：{'✅' if result.success else '❌'} | {result.stop_reason}\n")

    # ── 第四步：知识库里没有的问题 ───────────────────────────────
    print("=" * 60)
    print("第4步：知识库里没有答案时自动降级")
    print("=" * 60)

    result = agent.run("根据我的知识库，Mnemis 的前端 UI 用了什么框架？")
    print(f"\n状态：{'✅' if result.success else '❌'} | {result.stop_reason}\n")

    # ── 第五步：列出知识库内容 ───────────────────────────────────
    print("=" * 60)
    print("第5步：查看知识库里有什么")
    print("=" * 60)

    result = agent.run("告诉我知识库里现在有哪些内容")
    print(f"\n状态：{'✅' if result.success else '❌'} | {result.stop_reason}\n")


if __name__ == "__main__":
    main()
