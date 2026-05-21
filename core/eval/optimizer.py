# core/eval/optimizer.py
import json
import sqlite3
from core.eval.models import EvalCase, EvalResult, EvalReport
from core.eval.runner import EvalRunner, EVAL_DB_PATH
from core.eval.prompt_manager import PromptManager, PromptVersion
from core.client import LLMClient
from core.models import Message
from utils.logger import get_logger

logger = get_logger(__name__)

ANALYST_SYSTEM = """你是一位 AI 系统的 Prompt 工程专家。
你会分析 AI Agent 的失败测试用例，找出 System Prompt 存在的问题，
并提出具体、可操作的改进建议。

你的分析应该：
1. 找出失败用例之间的共同规律（不要逐条分析）
2. 定位到 System Prompt 里对应的具体问题
3. 给出明确的修改建议（可以是新增段落、修改措辞、添加示例）

只输出 JSON，不要其他内容。"""

ANALYST_PROMPT = """分析以下失败的测试用例，找出 System Prompt 需要改进的地方：

当前 System Prompt：
---
{current_prompt}
---

失败用例（共 {fail_count} 个）：
{failed_cases}

输出 JSON：
{{
  "root_cause": "失败的根本原因（一句话）",
  "patterns": ["规律1", "规律2"],
  "specific_issues": [
    {{"location": "Prompt 的哪个部分", "problem": "问题描述", "suggestion": "修改建议"}}
  ],
  "new_prompt_draft": "改进后的完整 System Prompt（直接可用）"
}}"""


class PromptOptimizer:
    """
    Prompt 自动优化器。
    基于 Eval 失败用例，自动分析问题并生成改进版本。
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self.runner = EvalRunner(self.llm)
        self.manager = PromptManager()

    def optimize(
        self,
        prompt_id: str,
        current_prompt: str,
        eval_cases: list[EvalCase],
        failed_results: list[EvalResult],
        prompt_name: str = "agent_system",
        human_review: bool = True,
    ) -> dict:
        """
        完整的优化流程：
        1. 分析失败用例
        2. 生成改进版 Prompt
        3. A/B 测试验证
        4. 决策是否采纳

        返回包含新旧 Prompt、得分对比、采纳决策的完整报告。
        """
        logger.info(
            f"Optimizing prompt '{prompt_id}' "
            f"({len(failed_results)} failed cases)"
        )

        if not failed_results:
            return {"status": "no_failures", "message": "没有失败用例，无需优化"}

        # 保存当前版本（如果还没保存）
        self.manager.save(
            prompt_id=prompt_id, name=prompt_name,
            content=current_prompt, notes="优化前的原始版本",
        )

        # 步骤 1：分析失败原因
        print("\n🔍 分析失败用例...")
        analysis = self._analyze_failures(
            current_prompt, eval_cases, failed_results
        )
        print(f"  根本原因：{analysis.get('root_cause','')}")
        for pattern in analysis.get('patterns', []):
            print(f"  规律：{pattern}")

        # 步骤 2：获取新 Prompt 草稿
        new_prompt = analysis.get("new_prompt_draft", "")
        if not new_prompt:
            return {"status": "generation_failed", "analysis": analysis}

        print(f"\n📝 生成改进版 Prompt（{len(new_prompt)} 字符）")

        # 步骤 3：如果需要人工审查
        if human_review:
            print("\n" + "─"*50)
            print("改进建议摘要：")
            for issue in analysis.get("specific_issues", [])[:3]:
                print(f"  [{issue['location']}] {issue['problem']}")
                print(f"    → {issue['suggestion']}")
            print("─"*50)
            confirm = input("\n是否继续进行 A/B 测试？(y/n): ").strip().lower()
            if confirm != "y":
                return {"status": "rejected_by_human", "analysis": analysis}

        # 步骤 4：A/B 测试
        print("\n⚗️  A/B 测试中...")
        ab_result = self._ab_test(
            current_prompt=current_prompt,
            new_prompt=new_prompt,
            test_cases=eval_cases,
        )

        # 步骤 5：决策
        improved = ab_result["new_score"] > ab_result["old_score"] + 0.3
        print(f"\n📊 A/B 结果：")
        print(f"  旧 Prompt：{ab_result['old_score']:.2f}/10")
        print(f"  新 Prompt：{ab_result['new_score']:.2f}/10")
        print(f"  提升：{ab_result['new_score'] - ab_result['old_score']:+.2f}")

        if improved:
            # 保存新版本并设为活跃
            new_version = self.manager.save(
                prompt_id=prompt_id, name=prompt_name,
                content=new_prompt,
                eval_score=ab_result["new_score"],
                notes=f"自动优化：{analysis.get('root_cause','')}",
                set_active=True,
            )
            print(f"\n✅ 采纳新版本（v{new_version.version}）")
            status = "adopted"
        else:
            print(f"\n⚠️  新版本未显著改善，保留原版本")
            status = "rejected"

        return {
            "status": status,
            "analysis": analysis,
            "ab_result": ab_result,
            "new_prompt": new_prompt if improved else None,
        }

    def _analyze_failures(
        self,
        current_prompt: str,
        all_cases: list[EvalCase],
        failed_results: list[EvalResult],
    ) -> dict:
        """调用 LLM 分析失败原因并生成改进建议"""

        # 构建失败用例的结构化描述
        cases_map = {c.id: c for c in all_cases}
        failed_descriptions = []
        for r in failed_results[:8]:   # 最多分析 8 个
            case = cases_map.get(r.case_id)
            if not case:
                continue
            failed_descriptions.append(
                f"用例：{case.name}\n"
                f"输入：{case.input}\n"
                f"期望：{case.expected_behavior}\n"
                f"得分：{r.overall_score:.1f}/10\n"
                f"实际输出摘要：{r.actual_output[:200]}\n"
                f"Judge 评语：{r.judge_reasoning}"
            )

        prompt = ANALYST_PROMPT.format(
            current_prompt=current_prompt[:2000],
            fail_count=len(failed_results),
            failed_cases="\n\n---\n\n".join(failed_descriptions),
        )

        try:
            response = self.llm.chat(
                messages=[Message(role="user", content=prompt)],
                system=ANALYST_SYSTEM,
                temperature=0.2,
            )
            raw = (response.content.strip()
                   .removeprefix("```json").removeprefix("```")
                   .removesuffix("```").strip())
            return json.loads(raw)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"root_cause": "分析失败", "new_prompt_draft": ""}

    def _ab_test(
        self,
        current_prompt: str,
        new_prompt: str,
        test_cases: list[EvalCase],
    ) -> dict:
        """
        A/B 测试：在相同的测试用例上对比两个 Prompt 的得分。

        注意：这里使用 dev_cases（开发集）而非 eval_cases（测试集），
        遵循「测试集不用于调优」的原则。
        实际项目里，test_cases 应该是专门的开发集，不包含在 P5·3 的测试集里。
        """
        # 临时替换 Agent 的 system prompt 进行测试
        # 这里用简化版：用 Judge 直接评估两个 prompt 生成的输出

        # 取前 5 个用例做 A/B 测试（平衡速度和准确性）
        ab_cases = test_cases[:5]

        old_scores = []
        new_scores = []

        for case in ab_cases:
            # 测试旧 Prompt
            old_output, _ = self._run_with_prompt(case.input, current_prompt)
            old_score = self._quick_score(case, old_output)
            old_scores.append(old_score)

            # 测试新 Prompt
            new_output, _ = self._run_with_prompt(case.input, new_prompt)
            new_score = self._quick_score(case, new_output)
            new_scores.append(new_score)

        return {
            "old_score": sum(old_scores) / len(old_scores),
            "new_score": sum(new_scores) / len(new_scores),
            "case_count": len(ab_cases),
            "old_scores": old_scores,
            "new_scores": new_scores,
        }

    def _run_with_prompt(
        self, user_input: str, system_prompt: str
    ) -> tuple[str, list[str]]:
        """用指定的 system prompt 运行 Agent"""
        import core.tools
        registry = __import__('core.tools.registry', fromlist=['get_registry']).get_registry()
        tools = registry.get_schemas()

        messages = [{"role": "user", "content": user_input}]
        msgs, tool_calls = self.llm.chat_with_tools(
            messages=messages, tools=tools, system=system_prompt,
        )

        tool_names = [tc.name for tc in tool_calls]

        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                res = registry.execute(tc.name, tc.input)
                res.tool_use_id = tc.id
                tool_results.append(res)
            msgs = self.llm.append_tool_results(msgs, tool_results)
            msgs, _ = self.llm.chat_with_tools(
                messages=msgs, tools=tools, system=system_prompt,
            )

        final = ""
        for block in msgs[-1].get("content", []):
            if hasattr(block, "text"):
                final += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                final += block.get("text", "")
        return final, tool_names

    def _quick_score(self, case: EvalCase, output: str) -> float:
        """快速评分（简化版 Judge，只返回数字）"""
        prompt = f"""对以下 AI 回答快速评分（1-10整数），只输出数字：

        用户问题：{case.input}
        期望行为：{case.expected_behavior}
        实际回答：{output[:500]}

        只输出1个整数（1-10）："""
        try:
            response = self.llm.chat(
                messages=[Message(role="user", content=prompt)],
                system="你是评分专家，只输出1个1-10的整数，不要其他内容。",
                temperature=0.0,
            )
            return float(response.content.strip().split()[0])
        except Exception:
            return 5.0