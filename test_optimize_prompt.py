# optimize_demo.py
import core.tools
from core.client import LLMClient
from core.eval.cases import MNEMIS_EVAL_CASES
from core.eval.runner import EvalRunner
from core.eval.reporter import EvalReporter
from core.eval.optimizer import PromptOptimizer
from core.eval.prompt_manager import PromptManager

# ── 当前 Mnemis 的 System Prompt（待优化对象）────────────────────
CURRENT_SYSTEM_PROMPT = """你是 Mnemis，一个 AI 研究助手。
你有工具可以搜索网络、查询知识库和访问用户记忆。
请根据用户需求提供帮助。"""

# ── 一个特意写得有明显问题的 System Prompt（用于演示优化效果）────
WEAK_SYSTEM_PROMPT = """你是一个 AI 助手，请帮助用户。"""


def demo_full_optimization_cycle():
    """演示完整的优化循环"""
    print("=" * 60)
    print("Prompt 自动优化演示")
    print("=" * 60)

    llm = LLMClient()
    runner = EvalRunner(llm)
    reporter = EvalReporter()
    optimizer = PromptOptimizer(llm)
    manager = PromptManager()

    prompt_id = "mnemis_system"

    # ── 步骤1：保存初始 Prompt 版本 ──────────────────────────────
    print("\n📌 步骤1：保存初始 Prompt 版本")
    v1 = manager.save(
        prompt_id=prompt_id,
        name="Mnemis System Prompt",
        content=WEAK_SYSTEM_PROMPT,    # 故意用差的版本做演示
        notes="初始版本（待优化）",
        set_active=True,
    )
    print(f"  已保存 v{v1.version}（hash: {v1.content_hash}）")

    # ── 步骤2：运行 Eval，找出失败用例 ───────────────────────────
    print("\n📊 步骤2：运行 Eval（使用弱 Prompt）")

    # 选取 6 个有代表性的用例做演示
    demo_cases = [
        c for c in MNEMIS_EVAL_CASES
        if c.id in ["tc-001", "tc-002", "qa-001", "fa-001", "if-001", "comp-001"]
    ]

    report_v1 = runner.run(
        cases=demo_cases,
        agent_version=f"v{v1.version}-weak",
        notes="使用弱 System Prompt 的基准",
        verbose=True,
    )

    failed = [r for r in report_v1.results if not r.passed]
    print(f"\n  结果：通过率 {report_v1.pass_rate:.1%}，失败 {len(failed)} 个用例")

    if not failed:
        print("  所有用例通过，无需优化（换用 WEAK_SYSTEM_PROMPT 再试）")
        return

    # ── 步骤3：自动优化 ──────────────────────────────────────────
    print(f"\n⚡ 步骤3：自动优化（基于 {len(failed)} 个失败用例）")

    result = optimizer.optimize(
        prompt_id=prompt_id,
        current_prompt=WEAK_SYSTEM_PROMPT,
        eval_cases=demo_cases,
        failed_results=failed,
        prompt_name="Mnemis System Prompt",
        human_review=True,   # 改为 False 可跳过人工确认
    )

    if result["status"] == "rejected_by_human":
        print("  优化被人工拒绝")
        return

    if result["status"] != "adopted":
        print(f"  优化未采纳：{result['status']}")
        return

    # ── 步骤4：用新 Prompt 再次运行 Eval，验证改进效果 ──────────
    new_prompt = result["new_prompt"]
    print(f"\n✅ 步骤4：验证新 Prompt 效果")

    report_v2 = runner.run(
        cases=demo_cases,
        agent_version=f"v2-optimized",
        notes="自动优化后的版本",
        verbose=True,
    )

    # ── 步骤5：对比报告 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("优化前后对比")
    print(f"{'='*60}")
    print(f"  旧版本通过率：{report_v1.pass_rate:.1%}，平均分：{report_v1.avg_score:.2f}")
    print(f"  新版本通过率：{report_v2.pass_rate:.1%}，平均分：{report_v2.avg_score:.2f}")
    delta = report_v2.avg_score - report_v1.avg_score
    print(f"  提升：{delta:+.2f}分 ({'✅ 改善' if delta > 0 else '⚠️ 退步'})")


def demo_version_management():
    """演示版本管理和回滚"""
    print("\n" + "=" * 60)
    print("Prompt 版本管理演示")
    print("=" * 60)

    manager = PromptManager()
    prompt_id = "mnemis_system"

    # 列出所有历史版本
    versions = manager.list_versions(prompt_id)

    if not versions:
        print("  暂无版本记录（先运行 demo_full_optimization_cycle）")
        return

    print(f"\n历史版本（共 {len(versions)} 个）：\n")
    print(f"  {'版本':<8} {'评分':<8} {'状态':<10} {'备注'}")
    print("  " + "-"*55)
    for v in versions:
        score_str = f"{v.eval_score:.2f}" if v.eval_score else "  N/A"
        active_str = "★ 活跃" if v.is_active else "  历史"
        print(f"  v{v.version:<7} {score_str:<8} {active_str:<10} {v.notes[:30]}")

    # 演示回滚
    if len(versions) >= 2:
        target_version = versions[-1].version  # 回滚到最老版本
        print(f"\n演示：回滚到 v{target_version}...")
        success = manager.rollback(prompt_id, target_version)
        print(f"  {'✅ 回滚成功' if success else '❌ 回滚失败'}")

        active = manager.get_active(prompt_id)
        print(f"  当前活跃版本：v{active.version if active else 'N/A'}")


def demo_dimension_targeted_optimization():
    """演示针对特定薄弱维度的定向优化"""
    print("\n" + "=" * 60)
    print("定向优化演示：只针对「工具调用准确性」薄弱点")
    print("=" * 60)

    from core.eval.cases import get_cases_by_dimension
    from core.eval.models import EvalDimension

    llm = LLMClient()
    runner = EvalRunner(llm)

    # 只跑工具调用相关的用例
    tool_cases = get_cases_by_dimension(EvalDimension.TOOL_SELECTION)
    print(f"  工具调用相关用例：{len(tool_cases)} 个")

    report = runner.run(
        cases=tool_cases,
        agent_version="tool-eval",
        notes="工具调用专项测试",
        verbose=True,
    )

    failed = [r for r in report.results if not r.passed]
    print(f"\n  工具调用通过率：{report.pass_rate:.1%}")

    if failed:
        print(f"  失败用例：")
        cases_map = {c.id: c for c in tool_cases}
        for r in failed:
            case = cases_map.get(r.case_id)
            if case:
                print(f"    [{r.overall_score:.1f}] {case.name}")
                print(f"      调用了：{r.tool_calls_made or '无工具'}")
                print(f"      期望：{case.expected_behavior[:60]}...")


if __name__ == "__main__":
    demo_full_optimization_cycle()
    demo_version_management()
    demo_dimension_targeted_optimization()