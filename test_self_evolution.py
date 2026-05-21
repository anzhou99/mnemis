# evolution_demo.py
import asyncio
import core.tools
from core.client import LLMClient
from core.planning.planner import GoalPlanner
from core.planning.executor import PlanExecutor
from core.planning.evolution import EvolutionLoop, QualityMonitor
from core.eval.cases import MNEMIS_EVAL_CASES, get_cases_by_tag


def demo_quality_monitoring():
    """演示质量监控触发机制"""
    print("=" * 60)
    print("演示1：质量监控——模拟质量下滑触发演进")
    print("=" * 60)

    monitor = QualityMonitor(window_size=10)

    # 模拟一段时间的评分：先正常，后下滑
    score_sequence = [8.2, 7.8, 8.0, 7.5, 7.2, 6.8, 6.2, 5.9, 5.5, 5.8]
    print("\n模拟评分序列（最近10次执行）：")
    for i, score in enumerate(score_sequence, 1):
        monitor.record_score(score)
        avg = monitor.rolling_avg
        trend = monitor.trend
        trigger = "⚠️ 触发演进！" if monitor.should_trigger_evolution else ""
        if avg:
            print(f"  第{i:2d}次：{score:.1f}  滚动均分={avg:.2f}  趋势={trend} {trigger}")
        else:
            print(f"  第{i:2d}次：{score:.1f}  （样本积累中...）")


async def demo_full_evolution():
    """演示完整的自我演进闭环（快速版，只跑4个测试用例）"""
    print("\n" + "=" * 60)
    print("演示2：完整演进闭环（手动触发）")
    print("=" * 60)

    evolution = EvolutionLoop(
        human_review=True,    # 改为 False 可全自动运行
        auto_rollback=True,
    )

    # 打印演进前的状态
    evolution.print_status()

    # 使用4个用例的快速测试集（节省时间）
    quick_cases = [
        c for c in MNEMIS_EVAL_CASES
        if c.id in ["tc-001", "qa-001", "fa-001", "if-001"]
    ]

    print(f"\n手动触发演进（{len(quick_cases)} 个测试用例）...")
    record = evolution.evolve(
        triggered_by="manual",
        cases=quick_cases,
        verbose=True,
    )

    if record:
        print(f"\n演进记录：")
        print(f"  v{record.old_version} → v{record.new_version}")
        print(f"  得分：{record.baseline_score:.2f} → {record.new_score:.2f}")
        print(f"  原因：{record.notes[:80]}")
    else:
        print("\n本次演进未采纳新版本")

    # 打印演进后的状态
    evolution.print_status()


def demo_integrated_with_planner():
    """演示演进监控与 PlanExecutor 的集成"""
    print("\n" + "=" * 60)
    print("演示3：PlanExecutor + 演进监控集成")
    print("=" * 60)

    llm = LLMClient()
    evolution = EvolutionLoop(human_review=False, auto_rollback=True)
    planner = GoalPlanner(llm)
    executor = PlanExecutor(
        planner=planner,
        llm=llm,
        verbose=True,
        enable_reflection=True,
        write_to_memory=True,
        evolution_loop=evolution,    # 传入演进监控器
    )

    async def run():
        plan = planner.plan(
            "搜索 Python 3.13 的两个核心新特性，整理成简洁的技术笔记",
            context={"format": "Markdown，每个特性200字以内"}
        )
        plan = await executor.execute(plan)

        # 检查是否积累了足够分数来显示趋势
        avg = evolution.monitor.rolling_avg
        if avg:
            print(f"\n📊 执行后质量监控：均分={avg:.2f}，趋势={evolution.monitor.trend}")
        else:
            scores_so_far = len(evolution.monitor._scores)
            needed = evolution.monitor.window_size // 2
            print(f"\n📊 质量监控：已收集 {scores_so_far}/{needed} 个样本")

        print(f"\n执行总结：{plan.final_summary[:200]}...")

    asyncio.run(run())


def demo_rollback_protection():
    """演示自动回滚保护"""
    print("\n" + "=" * 60)
    print("演示4：回滚保护——新版本比旧版本差时自动回滚")
    print("=" * 60)

    from core.eval.prompt_manager import PromptManager

    manager = PromptManager()
    prompt_id = "mnemis_system"
    versions = manager.list_versions(prompt_id)

    if len(versions) < 2:
        print("  需要至少2个版本来演示回滚（先运行 demo_full_evolution）")
        return

    current = manager.get_active(prompt_id)
    print(f"当前活跃版本：v{current.version if current else '?'}")
    print("\n版本历史：")
    for v in versions[:4]:
        active = "★ 活跃" if v.is_active else "  历史"
        score = f"{v.eval_score:.2f}" if v.eval_score else "N/A"
        print(f"  {active} v{v.version} [{score}] {v.notes[:50]}")

    # 演示手动回滚
    if len(versions) >= 2:
        target = versions[1].version   # 回滚到倒数第二个版本
        print(f"\n演示：手动回滚到 v{target}")
        success = manager.rollback(prompt_id, target)
        active_after = manager.get_active(prompt_id)
        print(f"  {'✅ 回滚成功' if success else '❌ 回滚失败'}")
        print(f"  当前活跃版本：v{active_after.version if active_after else '?'}")


if __name__ == "__main__":
    # 演示1：质量监控机制（无 API 调用，快速）
    demo_quality_monitoring()

    # 演示2：完整演进闭环（会消耗较多 API 调用）
    asyncio.run(demo_full_evolution())

    # 演示3：与 PlanExecutor 集成
    demo_integrated_with_planner()

    # 演示4：回滚保护
    demo_rollback_protection()