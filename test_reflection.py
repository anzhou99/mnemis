# reflection_demo.py
import asyncio
import core.tools
from core.client import LLMClient
from core.planning.planner import GoalPlanner
from core.planning.executor import PlanExecutor
from core.planning.reflection import SelfCritic, ReflectionMemoryWriter
from core.planning.models import PlanStep, StepStatus


async def demo_reflection_in_action():
    """完整演示：带反思的计划执行"""
    print("=" * 60)
    print("演示：带自我反思的目标执行")
    print("=" * 60)

    llm = LLMClient()
    planner = GoalPlanner(llm)
    executor = PlanExecutor(
        planner=planner,
        llm=llm,
        verbose=True,
        enable_reflection=True,
        write_to_memory=True,
    )

    plan = planner.plan(
        "搜索 Python asyncio 的3个核心使用模式，整理成带代码示例的笔记保存到文件",
        context={"audience": "有 Python 基础的开发者"}
    )

    plan = await executor.execute(plan)

    print(f"\n{'='*60}")
    print(f"执行总结：{plan.progress_summary()}")

    # 展示各步骤的反思结果
    print("\n📊 各步骤质量评估：")
    for step in plan.steps:
        if step.quality_score is not None:
            bar = "█" * int(step.quality_score) + "░" * (10 - int(step.quality_score))
            print(f"  [{step.id}] {step.quality_score:.1f}/10 [{bar}]")
            if step.reflection:
                print(f"    改进建议：{step.reflection[:80]}...")

    print(f"\n最终总结：\n{plan.final_summary}")


def demo_standalone_critic():
    """单独演示 SelfCritic 的评估能力"""
    print("\n" + "=" * 60)
    print("单独演示：SelfCritic 评估不同质量的输出")
    print("=" * 60)

    llm = LLMClient()
    critic = SelfCritic(llm)

    test_cases = [
        {
            "description": "搜索 Python asyncio 的核心概念",
            "milestone": "找到至少 3 个核心概念，每个有简短说明",
            "result": """asyncio 核心概念：
1. 事件循环（Event Loop）：asyncio 的核心调度器，管理所有协程的执行
2. 协程（Coroutine）：用 async def 定义，可以被挂起和恢复的函数
3. Task：对协程的封装，让事件循环能调度它""",
            "label": "高质量输出"
        },
        {
            "description": "搜索 Python asyncio 的核心概念",
            "milestone": "找到至少 3 个核心概念，每个有简短说明",
            "result": "asyncio 是 Python 的异步库，很有用。",
            "label": "低质量输出"
        },
    ]

    for case in test_cases:
        step = PlanStep(
            id=f"test-{case['label']}",
            description=case["description"],
            milestone=case["milestone"],
        )
        step.result = case["result"]

        print(f"\n测试：{case['label']}")
        reflection = critic.reflect(step)
        bar = "█" * int(reflection.score) + "░" * (10 - int(reflection.score))
        print(f"  评分：{reflection.score:.1f}/10 [{bar}]")
        print(f"  达标：{'✓' if reflection.milestone_met else '✗'}")
        print(f"  失败类型：{reflection.failure_type or '无'}")
        print(f"  需重试：{reflection.should_retry}")
        if reflection.weaknesses:
            print(f"  不足：{', '.join(reflection.weaknesses[:2])}")
        if reflection.reusable_insight:
            print(f"  可复用洞察：{reflection.reusable_insight[:80]}")


def demo_memory_accumulation():
    """演示经验积累到记忆系统"""
    print("\n" + "=" * 60)
    print("演示：执行经验写入长期记忆")
    print("=" * 60)

    from core.memory.database import MemoryDatabase
    from core.memory.semantic import SemanticMemory
    from core.memory.models import MemoryType

    db = MemoryDatabase()
    llm = LLMClient()
    semantic = SemanticMemory(db, llm)

    # 查看已积累的 background 类经验记忆
    experiences = semantic.get_all_by_type(
        memory_type=MemoryType.BACKGROUND,
        min_confidence=0.5,
        limit=10,
    )

    print(f"当前积累的执行经验：{len(experiences)} 条")
    for exp in experiences:
        print(f"  [{exp.confidence:.1f}] {exp.content}")

    if not experiences:
        print("  （暂无积累的经验，先运行 demo_reflection_in_action）")

    # 演示：用语义检索找相关经验
    if experiences:
        query = "搜索任务如何做得更好"
        relevant = semantic.search_by_semantic(query, top_k=3)
        print(f"\n查询「{query}」相关经验：")
        for r in relevant:
            print(f"  {r.content}")


if __name__ == "__main__":
    asyncio.run(demo_reflection_in_action())
    demo_standalone_critic()
    demo_memory_accumulation()