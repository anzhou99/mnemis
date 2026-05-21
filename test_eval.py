# run_eval.py
import core.tools
from core.eval.cases import MNEMIS_EVAL_CASES, get_cases_by_tag
from core.eval.runner import EvalRunner
from core.eval.reporter import EvalReporter
from core.eval.models import EvalCase


def run_baseline():
    """建立基准 Eval，这是后续所有改进的参照点"""
    runner = EvalRunner()
    reporter = EvalReporter()

    # 建立用例 ID → 名称的映射，方便报告展示
    cases_map = {c.id: c.name for c in MNEMIS_EVAL_CASES}

    print("🚀 运行基准 Eval（全部 20 个用例）...")
    print("⚠️  这会消耗较多 API 调用（每个用例 2-3 次），约需 3-5 分钟\n")

    report = runner.run(
        cases=MNEMIS_EVAL_CASES,
        agent_version="v1.0-baseline",
        notes="P5·3 建立的初始基准",
        verbose=True,
    )

    reporter.print_report(report, cases_map)
    print(f"\n💾 报告已保存，运行 ID：{report.run_id}")
    return report.run_id


def run_quick_smoke():
    """快速冒烟测试：只跑关键用例，快速验证基本功能"""
    runner = EvalRunner()
    reporter = EvalReporter()

    # 选每个标签组的第一个用例作为快速测试
    quick_cases = [
        next(c for c in MNEMIS_EVAL_CASES if "tool_use" in c.tags),
        next(c for c in MNEMIS_EVAL_CASES if "qa" in c.tags),
        next(c for c in MNEMIS_EVAL_CASES if "hallucination" in c.tags),
        next(c for c in MNEMIS_EVAL_CASES if "format" in c.tags),
    ]

    print("⚡ 快速冒烟测试（4 个关键用例）...\n")
    report = runner.run(
        cases=quick_cases,
        agent_version="smoke-test",
        verbose=True,
    )
    reporter.print_report(report, {c.id: c.name for c in quick_cases})
    return report.run_id


def show_history():
    """显示历史 Eval 运行记录"""
    reporter = EvalReporter()
    history = reporter.get_history(limit=5)

    if not history:
        print("暂无历史 Eval 记录")
        return

    print("\n📈 历史 Eval 记录：\n")
    print(f"  {'日期':<16} {'版本':<20} {'通过率':>8} {'平均分':>8} {'用例数':>6}")
    print("  " + "-" * 62)
    for h in history:
        print(f"  {h['date']:<16} {h['version']:<20} {h['pass_rate']:>8} {h['avg_score']:>8} {h['total']:>6}")


if __name__ == "__main__":

    run_baseline()
    show_history()
    # run_quick_smoke()
    # show_history()