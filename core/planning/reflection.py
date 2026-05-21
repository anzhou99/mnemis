# core/planning/reflection.py
import json
from datetime import datetime
from dataclasses import dataclass, field
from core.client import LLMClient
from core.models import Message
from core.planning.models import PlanStep, StepStatus
from utils.logger import get_logger

logger = get_logger(__name__)

# 质量得分低于此阈值时触发重试
QUALITY_THRESHOLD = 6.0

CRITIC_SYSTEM = """你是一个严格、客观的执行质量评估专家。
你的任务是评估一个 AI Agent 完成某个步骤的质量。

评估维度：
1. 完整性：是否达到了里程碑要求的所有标准？
2. 准确性：结果内容是否准确可信？
3. 效率：过程是否简洁，有无冗余步骤？

评分标准：
9-10：超出预期，完美完成
7-8：良好，达到要求，有小的改进空间
5-6：及格，完成了主要目标但有明显不足
3-4：较差，部分完成但缺失重要内容
1-2：失败，基本未达到目标

只输出 JSON，不要其他内容。"""

CRITIC_PROMPT = """评估以下步骤的执行质量：

步骤描述：{description}
完成标准（里程碑）：{milestone}

实际执行结果：
{result}

输出 JSON 示例，保持结构一致：
{{
  "score": 7,
  "milestone_met": true,
  "failure_type": null,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "improvement_suggestion": "如果要做得更好，应该...",
  "reusable_insight": "这类任务的关键经验是..."
}}

failure_type 只在 milestone_met=false 时填写：
"tool_failure" / "planning_failure" / "understanding_failure"
"""

EXPERIENCE_EXTRACTION_PROMPT = """基于以下执行经验，提炼一条可以推广到类似任务的具体建议：

任务类型：{task_type}
执行情况：{summary}
关键洞察：{insight}

输出一条简洁、可操作的经验（50字以内），格式：「在做[任务类型]时，[具体建议]」"""


@dataclass
class ReflectionResult:
    """自我反思的完整结果"""

    step_id: str
    score: float  # 1-10
    milestone_met: bool
    failure_type: str | None = None  # tool/planning/understanding/None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improvement_suggestion: str = ""
    reusable_insight: str = ""  # 可写入记忆的经验
    should_retry: bool = False
    reflected_at: datetime = field(default_factory=datetime.now)

    @property
    def needs_replan(self) -> bool:
        """理解失败时需要触发重规划，而不仅仅是重试"""
        return self.failure_type == "understanding_failure"


class SelfCritic:
    """
    自我批评器：对步骤执行结果进行结构化评估。
    三个核心能力：
    1. 评估质量得分（决定是否重试）
    2. 识别失败类型（决定应对策略）
    3. 提炼可复用经验（写入长期记忆）
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def reflect(self, step: PlanStep) -> ReflectionResult:
        """
        对一个步骤的执行结果进行完整的三层反思。
        """
        if not step.result:
            return ReflectionResult(
                step_id=step.id,
                score=0.0,
                milestone_met=False,
                failure_type="tool_failure",
                should_retry=True,
            )

        logger.debug(f"Reflecting on step {step.id}...")

        raw = self._call_critic(step)
        result = self._parse_critic_response(step.id, raw)

        # 低分时标记需要重试
        result.should_retry = (
            not result.milestone_met
            and result.score < QUALITY_THRESHOLD
            and result.failure_type != "understanding_failure"
        )

        # 把反思结果存回步骤
        step.reflection = result.improvement_suggestion
        step.quality_score = result.score

        logger.info(
            f"Reflection [{step.id}]: score={result.score} "
            f"milestone={'✓' if result.milestone_met else '✗'} "
            f"retry={result.should_retry}"
        )
        return result

    def _call_critic(self, step: PlanStep) -> dict:
        """调用 LLM 进行评估"""
        try:
            response = self.llm.chat(
                messages=[
                    Message(
                        role="user",
                        content=CRITIC_PROMPT.format(
                            description=step.description,
                            milestone=step.milestone,
                            result=step.result[:1500],  # 截断避免超出 context
                        ),
                    )
                ],
                system=CRITIC_SYSTEM,
                temperature=0.1,  # 评估任务用低温度，保持一致性
            )
            raw = (
                response.content.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Critic call failed: {e}")
            return {"score": 5, "milestone_met": True, "failure_type": None}

    def _parse_critic_response(self, step_id: str, raw: dict) -> ReflectionResult:
        score = float(raw.get("score", 5))
        score = max(1.0, min(10.0, score))  # 限制在 1-10

        return ReflectionResult(
            step_id=step_id,
            score=score,
            milestone_met=bool(raw.get("milestone_met", True)),
            failure_type=raw.get("failure_type"),
            strengths=raw.get("strengths", []),
            weaknesses=raw.get("weaknesses", []),
            improvement_suggestion=raw.get("improvement_suggestion", ""),
            reusable_insight=raw.get("reusable_insight", ""),
        )

    def extract_experience(
        self,
        step: PlanStep,
        reflection: ReflectionResult,
        task_context: str = "",
    ) -> str | None:
        """
        从反思结果里提炼一条可复用的经验。
        只有当洞察有足够价值时才提炼（避免存储无意义的经验）。
        """
        if not reflection.reusable_insight or reflection.score < 5:
            return None

        try:
            response = self.llm.chat(
                messages=[
                    Message(
                        role="user",
                        content=EXPERIENCE_EXTRACTION_PROMPT.format(
                            task_type=step.description[:100],
                            summary=f"得分 {reflection.score}/10，{reflection.improvement_suggestion[:200]}",
                            insight=reflection.reusable_insight[:300],
                        ),
                    )
                ],
                system="你是经验提炼专家，把具体案例提炼为可推广的通用建议。只输出一条建议，不超过50字。",
                temperature=0.2,
            )
            experience = response.content.strip()
            logger.debug(f"Experience extracted: {experience}")
            return experience
        except Exception as e:
            logger.error(f"Experience extraction failed: {e}")
            return None


class ReflectionMemoryWriter:
    """
    把反思结果写入长期记忆系统（P3 的 SemanticMemory）。
    让每次执行经验都能积累成 Agent 的「专业知识」。
    """

    def __init__(self):
        # 懒加载，避免在没有数据库时启动失败
        self._semantic = None
        self._db = None
        self._embedding = None

    def _get_semantic(self):
        if self._semantic is None:
            from core.memory.database import MemoryDatabase
            from core.memory.semantic import SemanticMemory
            from core.client import LLMClient
            from core.embeddings import Embeddings

            self._db = MemoryDatabase()
            self._semantic = SemanticMemory(self._db, LLMClient())
            self._embedding = Embeddings()
        return self._semantic

    def write_experience(
        self,
        experience_text: str,
        source_step_id: str,
        session_id: str | None = None,
    ):
        """
        把一条提炼出的经验写入语义记忆。
        使用 MemoryType.BACKGROUND 类型，表示这是 Agent 的「工作经验」。
        """
        try:
            from core.memory.models import Memory, MemoryType
            import uuid

            semantic = self._get_semantic()

            memory = Memory(
                id=str(uuid.uuid4()),
                content=experience_text,
                memory_type=MemoryType.BACKGROUND,  # 工作经验归类为 background
                confidence=0.8,  # 单次经验置信度中等，多次验证后会提高
                source_session_id=session_id,
            )

            # 向量化并写入（会经过 P3·5 的冲突检测）
            vector = self._embedding.embed_text(experience_text)

            import uuid as _uuid

            vector_id = str(_uuid.uuid4())
            from qdrant_client.models import PointStruct
            from core.memory.semantic import MEMORY_COLLECTION

            semantic._qdrant.upsert(
                collection_name=MEMORY_COLLECTION,
                points=[
                    PointStruct(
                        id=vector_id,
                        vector=vector,
                        payload={
                            "memory_id": memory.id,
                            "content": experience_text,
                            "memory_type": memory.memory_type.value,
                            "confidence": memory.confidence,
                            "session_id": session_id or "",
                            "source": f"reflection:{source_step_id}",
                        },
                    )
                ],
            )
            memory.vector_id = vector_id
            self._db.save_memory(memory)

            logger.info(f"Experience written to memory: {experience_text[:60]}...")

        except Exception as e:
            # 记忆写入失败不应影响主流程
            logger.error(f"Failed to write experience to memory: {e}")
