# core/eval/reporter.py
import sqlite3
import json
from pathlib import Path
from core.eval.models import EvalReport
from core.eval.runner import EVAL_DB_PATH


class EvalReporter:
    """生成 Eval 分析报告，支持历史趋势对比"""

    def print_report(self, report: EvalReport, cases_map: dict | None = None):
        """打印完整的评估报告"""
        print(f"\n{'='*60}")
        print(f"📊 Eval 报告  |  运行 ID：{report.run_id[:12]}...")
        print(f"   版本：{report.agent_version}  |  时间：{report.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")

        # 总览
        pass_icon = "🟢" if report.pass_rate >= 0.8 else "🟡" if report.pass_rate >= 0.6 else "🔴"
        print(f"{pass_icon} 通过率：{report.pass_rate:.1%}  ({report.passed_count}/{report.total_cases})")
        print(f"   平均分：{report.avg_score:.2f} / 10.00\n")

        # 按维度分析
        dim_scores = report.scores_by_dimension
        if dim_scores:
            print("📐 各维度得分：")
            for dim, score in sorted(dim_scores.items(), key=lambda x: x[1]):
                bar = "█" * int(score) + "░" * (10 - int(score))
                icon = "🔴" if score < 6 else "🟡" if score < 8 else "🟢"
                print(f"  {icon} {dim:<25} {score:.1f} [{bar}]")

        # 失败用例明细
        failed = [r for r in report.results if not r.passed and not r.failed]
        if failed:
            print(f"\n⚠️  未通过用例（{len(failed)} 个）：")
            for r in sorted(failed, key=lambda x: x.overall_score):
                case_name = cases_map.get(r.case_id, r.case_id) if cases_map else r.case_id
                print(f"  [{r.overall_score:.1f}] {case_name}")
                if r.judge_reasoning:
                    print(f"        {r.judge_reasoning[:80]}...")

        # 错误用例
        errors = [r for r in report.results if r.failed]
        if errors:
            print(f"\n❌ 执行错误（{len(errors)} 个）：")
            for r in errors:
                print(f"  [{r.case_id}] {r.error[:60]}")

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """对比两次 Eval 运行的结果"""
        with sqlite3.connect(EVAL_DB_PATH) as conn:
            def get_run(run_id):
                row = conn.execute(
                    "SELECT * FROM eval_runs WHERE run_id=?", (run_id,)
                ).fetchone()
                if not row:
                    return None
                cols = [d[0] for d in conn.description]
                return dict(zip(cols, row))

            run_a = get_run(run_id_a)
            run_b = get_run(run_id_b)

        if not run_a or not run_b:
            return {"error": "Run not found"}

        delta_pass = run_b["pass_rate"] - run_a["pass_rate"]
        delta_score = run_b["avg_score"] - run_a["avg_score"]

        return {
            "run_a": {"id": run_id_a[:12], **run_a},
            "run_b": {"id": run_id_b[:12], **run_b},
            "delta_pass_rate": delta_pass,
            "delta_avg_score": delta_score,
            "improved": delta_score > 0,
        }

    def get_history(self, limit: int = 10) -> list[dict]:
        """获取历史 Eval 运行记录"""
        with sqlite3.connect(EVAL_DB_PATH) as conn:
            rows = conn.execute(
                """SELECT run_id, agent_version, pass_rate, avg_score,
                          total_cases, created_at
                   FROM eval_runs
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [
            {
                "run_id": r[0][:12] + "...",
                "version": r[1],
                "pass_rate": f"{r[2]:.1%}",
                "avg_score": f"{r[3]:.2f}",
                "total": r[4],
                "date": r[5][:16],
            }
            for r in rows
        ]