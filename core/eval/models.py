# core/eval/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class EvalDimension(str, Enum):
    TOOL_SELECTION = "tool_selection"  # 工具选择准确性
    ANSWER_QUALITY = "answer_quality"  # 回答质量（相关性、深度）
    FACTUAL_ACCURACY = "factual_accuracy"  # 事实准确性
    INSTRUCTION_FOLLOW = "instruction_follow"  # 指令遵循度
    HALLUCINATION_FREE = "hallucination_free"  # 拒绝幻觉能力


@dataclass
class EvalCase:
    """
    单个评估测试用例。
    包含：输入、期望标准、评分维度、参考答案（可选）。
    """

    id: str = field(default_factory=lambda: f"case-{uuid.uuid4().hex[:6]}")
    name: str = ""
    input: str = ""  # 发给 Agent 的输入
    expected_behavior: str = ""  # 期望的行为描述（给 Judge 看的）
    reference_answer: str | None = None  # 参考答案（可选，提高评估一致性）
    dimensions: list[EvalDimension] = field(
        default_factory=lambda: [EvalDimension.ANSWER_QUALITY]
    )
    tags: list[str] = field(default_factory=list)  # 便于分组分析
    weight: float = 1.0  # 在总分中的权重


@dataclass
class EvalResult:
    """单个测试用例的评估结果"""

    case_id: str
    run_id: str
    actual_output: str = ""
    scores: dict[str, float] = field(default_factory=dict)  # 维度→分数
    overall_score: float = 0.0
    passed: bool = False  # 是否通过（overall ≥ threshold）
    judge_reasoning: str = ""  # Judge 的评估理由
    tool_calls_made: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    error: str = ""
    evaluated_at: datetime = field(default_factory=datetime.now)

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass
class EvalReport:
    """完整的一次 Eval 运行报告"""

    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    results: list[EvalResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    agent_version: str = "unknown"
    notes: str = ""

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_count / self.total_cases

    @property
    def avg_score(self) -> float:
        valid = [r.overall_score for r in self.results if not r.failed]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def scores_by_dimension(self) -> dict[str, float]:
        """各维度的平均分"""
        dim_scores: dict[str, list[float]] = {}
        for result in self.results:
            for dim, score in result.scores.items():
                dim_scores.setdefault(dim, []).append(score)
        return {dim: sum(scores) / len(scores) for dim, scores in dim_scores.items()}

    @property
    def scores_by_tag(self) -> dict[str, float]:
        """各标签的平均分（需要配合 cases 使用）"""
        return {}  # 在 Reporter 里实现
