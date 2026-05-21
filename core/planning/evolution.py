# core/planning/evolution.py
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from core.client import LLMClient
from core.eval.cases import MNEMIS_EVAL_CASES
from core.eval.runner import EvalRunner, EVAL_DB_PATH
from core.eval.reporter import EvalReporter
from core.eval.optimizer import PromptOptimizer
from core.eval.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

# ── 安全边界 ──────────────────────────────────────────────────────
MAX_EVOLUTION_PER_DAY    = 1
MAX_PROMPT_CHANGE_RATIO  = 0.4
MIN_IMPROVEMENT_THRESHOLD = 0.3
QUALITY_TRIGGER_THRESHOLD = 6.0
ROLLBACK_SCORE_DROP      = 1.0
ROLLING_WINDOW_SIZE      = 20   # 滚动窗口：最近 N 次 SelfCritic 评分


@dataclass
class EvolutionRecord:
    """单次演进记录"""
    id: str
    triggered_by: str          # "auto_quality" / "manual" / "scheduled"
    baseline_score: float      # 演进前的 Eval 平均分
    new_score: float           # 演进后的 Eval 平均分
    prompt_id: str
    old_version: int
    new_version: int
    adopted: bool
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class QualityMonitor:
    """
    质量监控器：追踪 SelfCritic 的滚动平均分，
    当质量持续下滑时触发演进。
    """

    def __init__(self, window_size: int = ROLLING_WINDOW_SIZE):
        self.window_size = window_size
        self._scores: list[float] = []

    def record_score(self, score: float):
        """记录一次 SelfCritic 评分"""
        self._scores.append(score)
        if len(self._scores) > self.window_size:
            self._scores.pop(0)

    @property
    def rolling_avg(self) -> float | None:
        if len(self._scores) < self.window_size // 2:
            return None  # 样本不足，不判断
        return sum(self._scores) / len(self._scores)

    @property
    def should_trigger_evolution(self) -> bool:
        avg = self.rolling_avg
        if avg is None:
            return False
        return avg < QUALITY_TRIGGER_THRESHOLD

    @property
    def trend(self) -> str:
        """返回质量趋势：improving / stable / declining"""
        if len(self._scores) < 6:
            return "unknown"
        recent = self._scores[-3:]
        older = self._scores[-6:-3]
        diff = sum(recent) / 3 - sum(older) / 3
        if diff > 0.3:
            return "improving"
        elif diff < -0.3:
            return "declining"
        return "stable"


class EvolutionLoop:
    """
    自主演进主循环。
    整合：质量监控 → Eval 触发 → Prompt 优化 → 验证 → 版本存档
    """

    PROMPT_ID = "mnemis_system"

    # Mnemis 的默认 System Prompt（演进的起点）
    DEFAULT_SYSTEM_PROMPT = """你是 Mnemis，一个拥有长期记忆和私有知识库的 AI 研究助手。

## 信息获取策略
根据问题性质选择合适的信息来源：
- 需要实时信息（新闻、价格、最新动态）→ 使用 web_search
- 涉及用户上传的文档或私有资料 → 使用 search_knowledge_base
- 用户提到「你还记得」「之前说过」→ 使用 recall_memories
- 通用知识和编程问题 → 直接回答，无需工具

## 回答规范
- 引用知识库内容时用 [数字] 标注来源
- 无法确定的信息主动说明，不要编造
- 代码示例要可运行，有简短注释

## 工具调用原则
按需调用，不过度调用。每次调用前确认：这个工具对当前问题真的必要吗？"""

    def __init__(
        self,
        human_review: bool = True,
        auto_rollback: bool = True,
    ):
        self.llm = LLMClient()
        self.monitor = QualityMonitor()
        self.runner = EvalRunner(self.llm)
        self.reporter = EvalReporter()
        self.optimizer = PromptOptimizer(self.llm)
        self.manager = PromptManager()
        self.human_review = human_review
        self.auto_rollback = auto_rollback
        self._evolution_history: list[EvolutionRecord] = []

        # 确保默认 Prompt 已保存
        existing = self.manager.get_active(self.PROMPT_ID)
        if not existing:
            self.manager.save(
                prompt_id=self.PROMPT_ID,
                name="Mnemis System Prompt",
                content=self.DEFAULT_SYSTEM_PROMPT,
                notes="初始版本",
                set_active=True,
            )

    # ── 核心：接收 SelfCritic 评分，决定是否触发演进 ─────────────

    def record_step_quality(self, score: float, step_description: str = ""):
        """
        每次 SelfCritic 完成评估后调用。
        积累足够样本后，自动判断是否需要演进。
        """
        self.monitor.record_score(score)
        avg = self.monitor.rolling_avg

        if avg is not None:
            logger.debug(
                f"Quality monitor: score={score:.1f} "
                f"rolling_avg={avg:.2f} trend={self.monitor.trend}"
            )

        # 触发条件：质量持续低于阈值，且今天还没触发过演进
        if (self.monitor.should_trigger_evolution
                and not self._evolved_today()):
            logger.warning(
                f"Quality below threshold ({avg:.2f} < {QUALITY_TRIGGER_THRESHOLD}), "
                f"triggering auto evolution"
            )
            self.evolve(triggered_by="auto_quality")

    def evolve(
        self,
        triggered_by: str = "manual",
        cases=None,
        verbose: bool = True,
    ) -> EvolutionRecord | None:
        """
        执行一次完整的演进流程。
        返回演进记录，如果没有采纳改进则返回 None。
        """
        if verbose:
            print(f"\n🔄 启动演进循环（触发原因：{triggered_by}）")
            if self.monitor.rolling_avg:
                print(f"   当前质量：{self.monitor.rolling_avg:.2f}/10，"
                      f"趋势：{self.monitor.trend}")

        eval_cases = cases or MNEMIS_EVAL_CASES
        current_pv = self.manager.get_active(self.PROMPT_ID)
        if not current_pv:
            logger.error("No active prompt found")
            return None

        # ── 步骤1：运行 Eval 获取基准分 ──────────────────────────
        if verbose:
            print("\n📊 运行 Eval 建立基准...")

        baseline_report = self.runner.run(
            cases=eval_cases,
            agent_version=f"before-evolution-v{current_pv.version}",
            notes=f"演进前基准（触发：{triggered_by}）",
            verbose=verbose,
        )
        baseline_score = baseline_report.avg_score

        failed_results = [r for r in baseline_report.results if not r.passed]
        if not failed_results:
            if verbose:
                print(f"✅ 所有用例通过（{baseline_score:.2f}/10），无需演进")
            return None

        if verbose:
            print(f"\n   基准分：{baseline_score:.2f}/10，{len(failed_results)} 个用例未通过")

        # ── 步骤2：执行 Prompt 优化 ───────────────────────────────
        if verbose:
            print("\n⚡ 执行 Prompt 优化...")

        opt_result = self.optimizer.optimize(
            prompt_id=self.PROMPT_ID,
            current_prompt=current_pv.content,
            eval_cases=eval_cases,
            failed_results=failed_results,
            human_review=self.human_review,
        )

        if opt_result["status"] != "adopted":
            if verbose:
                print(f"   优化未采纳（{opt_result['status']}），演进中止")
            return None

        # ── 步骤3：验证新版本 ─────────────────────────────────────
        new_pv = self.manager.get_active(self.PROMPT_ID)
        if not new_pv or new_pv.version == current_pv.version:
            return None

        if verbose:
            print(f"\n🧪 验证新版本（v{new_pv.version}）...")

        validation_report = self.runner.run(
            cases=eval_cases,
            agent_version=f"after-evolution-v{new_pv.version}",
            notes=f"演进后验证",
            verbose=verbose,
        )
        new_score = validation_report.avg_score

        # ── 步骤4：安全检查——如果变差了就自动回滚 ────────────────
        score_delta = new_score - baseline_score

        if self.auto_rollback and score_delta < -ROLLBACK_SCORE_DROP:
            if verbose:
                print(f"\n⚠️  新版本得分下降 {score_delta:.2f}，自动回滚！")
            self.manager.rollback(self.PROMPT_ID, current_pv.version)
            return None

        # ── 步骤5：记录演进结果 ───────────────────────────────────
        record = EvolutionRecord(
            id=f"evo-{int(time.time())}",
            triggered_by=triggered_by,
            baseline_score=baseline_score,
            new_score=new_score,
            prompt_id=self.PROMPT_ID,
            old_version=current_pv.version,
            new_version=new_pv.version,
            adopted=True,
            notes=opt_result.get("analysis", {}).get("root_cause", ""),
        )
        self._evolution_history.append(record)
        self._save_evolution_record(record)

        if verbose:
            delta_icon = "✅" if score_delta >= 0 else "⚠️"
            print(f"\n{delta_icon} 演进完成：{baseline_score:.2f} → {new_score:.2f} "
                  f"({score_delta:+.2f})")

        return record

    # ── 演进历史管理 ──────────────────────────────────────────────

    def _evolved_today(self) -> bool:
        """检查今天是否已经触发过演进"""
        today = datetime.now().date()
        return any(
            r.created_at.date() == today
            for r in self._evolution_history
        )

    def _save_evolution_record(self, record: EvolutionRecord):
        """把演进记录保存到数据库"""
        try:
            with sqlite3.connect(EVAL_DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS evolution_history (
                        id TEXT PRIMARY KEY,
                        triggered_by TEXT,
                        baseline_score REAL,
                        new_score REAL,
                        prompt_id TEXT,
                        old_version INTEGER,
                        new_version INTEGER,
                        adopted INTEGER,
                        notes TEXT,
                        created_at TEXT
                    )
                """)
                conn.execute(
                    "INSERT INTO evolution_history VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.id, record.triggered_by,
                        record.baseline_score, record.new_score,
                        record.prompt_id, record.old_version,
                        record.new_version, int(record.adopted),
                        record.notes, record.created_at.isoformat(),
                    )
                )
        except Exception as e:
            logger.error(f"Failed to save evolution record: {e}")

    def get_evolution_history(self, limit: int = 10) -> list[dict]:
        """获取演进历史"""
        try:
            with sqlite3.connect(EVAL_DB_PATH) as conn:
                rows = conn.execute(
                    """SELECT * FROM evolution_history
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
                cols = ["id","triggered_by","baseline_score","new_score",
                        "prompt_id","old_version","new_version","adopted","notes","created_at"]
                return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def print_status(self):
        """打印当前系统状态"""
        current_pv = self.manager.get_active(self.PROMPT_ID)
        versions = self.manager.list_versions(self.PROMPT_ID)
        history = self.get_evolution_history(limit=5)

        print(f"\n{'='*60}")
        print(f"🧬 Mnemis 演进状态")
        print(f"{'='*60}")

        # 当前版本
        if current_pv:
            print(f"\n活跃版本：v{current_pv.version}  "
                  f"(hash: {current_pv.content_hash})")
            score_str = f"{current_pv.eval_score:.2f}/10" if current_pv.eval_score else "未评估"
            print(f"评估得分：{score_str}")

        # 质量监控
        avg = self.monitor.rolling_avg
        if avg:
            bar = "█" * int(avg) + "░" * (10 - int(avg))
            print(f"\n质量监控：{avg:.2f}/10 [{bar}] 趋势：{self.monitor.trend}")
        else:
            print(f"\n质量监控：样本不足（需要 {self.monitor.window_size // 2} 条）")

        # 版本历史
        print(f"\n版本历史（共 {len(versions)} 个）：")
        for v in versions[:5]:
            active = "★" if v.is_active else " "
            score = f"{v.eval_score:.2f}" if v.eval_score else "N/A "
            print(f"  {active} v{v.version:<3} [{score}] {v.notes[:40]}")

        # 演进记录
        if history:
            print(f"\n演进记录（最近 {len(history)} 次）：")
            for h in history:
                delta = h["new_score"] - h["baseline_score"]
                icon = "✅" if delta >= 0 else "⚠️"
                print(f"  {icon} v{h['old_version']}→v{h['new_version']} "
                      f"{h['baseline_score']:.2f}→{h['new_score']:.2f} "
                      f"({delta:+.2f})  [{h['triggered_by']}]")