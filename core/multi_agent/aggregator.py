from core.multi_agent.models import AgentTask, AgentRole, TaskStatus, ResearchPlan
from core.multi_agent.context import SharedContext
from core.client import LLMClient
from core.models import Message
from utils.logger import get_logger

logger = get_logger(__name__)


class ResultAggregator:
    """
    结果聚合器：把多个 Sub-agent 的产出合并成最终报告。

    聚合策略：
    - 研究结果 → 直接传给写作者作为上下文
    - 写作结果 + 审核意见 → LLM 综合修改
    - 部分失败 → 输出已完成部分 + 注明缺失
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def aggregate(
        self,
        plan: ResearchPlan,
        context: SharedContext,
    ) -> str:
        """
        根据执行结果聚合最终报告。
        """
        done_tasks = [t for t in plan.tasks if t.status == TaskStatus.DONE]
        failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        skipped_tasks = [t for t in plan.tasks if t.status == TaskStatus.SKIPPED]

        logger.info(
            f"Aggregating: {len(done_tasks)} done, "
            f"{len(failed_tasks)} failed, {len(skipped_tasks)} skipped"
        )

        # 找到写作者的输出（主体内容）
        writer_results = context.get_results_by_role(
            AgentRole.REVIEWER.value, plan.tasks
        )

        # 找到审核者的意见
        reviewer_results = context.get_results_by_role(
            AgentRole.REVIEWER.value, plan.tasks
        )

        if not writer_results:
            # 写作任务失败，输出研究结果作为降级输出
            researcher_results = context.get_results_by_role(
                AgentRole.RESEARCHER.value, plan.tasks
            )

            if researcher_results:
                return self._format_partial_output(
                    plan.topic, researcher_results, "写作任务未完成，以下为原始研究结果"
                )
            return f"任务执行失败：{plan.topic}\n\n失败原因：\n" + "\n".join(
                f"- {t.role.value}: {t.error}" for t in failed_tasks
            )

        draft = writer_results[0]

        if not reviewer_results:
            # 没有审核结果，直接返回草稿
            logger.warning("No reviewer output, returning draft as-is")
            return draft

        # 有审核意见，让 LLM 整合修改
        return self._apply_review(draft, reviewer_results[0])

    def _apply_review(self, draft: str, review_notes: str) -> str:
        """根据审核意见修改草稿"""
        response = self.llm.chat(
            messages=[
                Message(
                    role="user",
                    content=f"""请根据以下审核意见修改文章草稿。

                    ## 草稿
                    {draft}

                    ## 审核意见
                    {review_notes}

                    请直接输出修改后的完整文章，不需要解释修改了什么。""",
                )
            ],
            system="你是专业的技术文章编辑，根据审核意见优化文章质量。保持原文的技术深度，提升表达的清晰度。",
            temperature=0.3,
        )
        return response.content

    def _format_partial_output(
        self,
        topic: str,
        results: list[str],
        note: str,
    ) -> str:
        return f"# {topic}\n\n> ⚠️ {note}\n\n" + "\n\n---\n\n".join(results)
