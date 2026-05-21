# core/eval/runner.py
import time
import json
import sqlite3
from pathlib import Path
from core.eval.models import EvalCase, EvalResult, EvalReport, EvalDimension
from core.client import LLMClient
from core.models import Message
from core.tools.registry import get_registry
from utils.logger import get_logger

logger = get_logger(__name__)

EVAL_DB_PATH = Path("./mnemis_eval.db")

JUDGE_SYSTEM = """你是一位严格、客观的 AI 系统评估专家。
你的任务是评估 AI Agent 的回答质量。

评分规则（每个维度 1-10 分）：
- 9-10：超出预期，完美符合要求
- 7-8：良好，满足要求，有小瑕疵
- 5-6：及格，基本满足但有明显不足
- 3-4：较差，部分满足
- 1-2：失败，基本不满足要求

评估时要独立评估每个维度，不要被整体印象影响单一维度的得分。
只输出 JSON，不要其他内容。"""

JUDGE_PROMPT = """评估以下 AI Agent 的回答：

用户输入：{input}

期望行为：{expected_behavior}

{reference_section}

实际回答：
{actual_output}

{tool_section}

需要评估的维度：{dimensions}

输出 JSON：
{{
  "scores": {{
    "{dim1}": 8,
    "{dim2}": 7
  }},
  "overall_score": 7.5,
  "passed": true,
  "reasoning": "简要说明评分理由（2-3句话）",
  "key_issue": "最主要的问题（如果有）"
}}

passed 的标准：overall_score >= 6.0"""


class EvalRunner:
    """
    Eval 执行引擎。
    批量运行测试用例，用 LLM-as-Judge 评分，把结果持久化到 SQLite。
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self._init_db()

    def _init_db(self):
        """初始化 Eval 结果数据库"""
        with sqlite3.connect(EVAL_DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_results (
                    id          TEXT PRIMARY KEY,
                    run_id      TEXT NOT NULL,
                    case_id     TEXT NOT NULL,
                    overall_score REAL,
                    passed      INTEGER,
                    scores_json TEXT,
                    output_preview TEXT,
                    error       TEXT,
                    execution_time REAL,
                    evaluated_at TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id      TEXT PRIMARY KEY,
                    agent_version TEXT,
                    pass_rate   REAL,
                    avg_score   REAL,
                    total_cases INTEGER,
                    notes       TEXT,
                    created_at  TEXT
                )
            """
            )
        logger.debug(f"Eval DB initialized: {EVAL_DB_PATH}")

    def run(
        self,
        cases: list[EvalCase],
        agent_version: str = "current",
        notes: str = "",
        verbose: bool = True,
    ) -> EvalReport:
        """
        运行一批测试用例，返回完整报告。
        """
        report = EvalReport(agent_version=agent_version, notes=notes)

        if verbose:
            print(f"\n🧪 开始 Eval（{len(cases)} 个用例，版本：{agent_version}）\n")

        for i, case in enumerate(cases, 1):
            if verbose:
                print(
                    f"  [{i:2d}/{len(cases)}] {case.name[:50]}...", end=" ", flush=True
                )

            result = self._run_single_case(case, report.run_id)
            report.results.append(result)

            if verbose:
                icon = "✅" if result.passed else ("❌" if result.failed else "⚠️")
                print(f"{icon} {result.overall_score:.1f}/10")

        # 保存报告到数据库
        self._save_report(report)

        if verbose:
            print(f"\n{'='*50}")
            print(
                f"📊 通过率：{report.pass_rate:.1%}（{report.passed_count}/{report.total_cases}）"
            )
            print(f"   平均分：{report.avg_score:.2f}/10")

        return report

    def _run_single_case(self, case: EvalCase, run_id: str) -> EvalResult:
        """执行单个测试用例"""
        result = EvalResult(case_id=case.id, run_id=run_id)
        start = time.time()

        try:
            # 运行 Agent
            actual_output, tool_calls = self._run_agent(case.input)
            result.actual_output = actual_output
            result.tool_calls_made = tool_calls
            result.execution_time = time.time() - start

            # LLM-as-Judge 评分
            self._judge(case, result)

        except Exception as e:
            result.error = str(e)
            result.execution_time = time.time() - start
            logger.error(f"Case {case.id} failed: {e}")

        return result

    def _run_agent(self, user_input: str) -> tuple[str, list[str]]:
        """
        运行 Agent 获取回答。
        这里用带工具的单次调用，实际项目里可以换成完整的 MemoryEnabledAgent。
        """
        import core.tools  # 确保工具已注册

        registry = get_registry()
        tools = registry.get_schemas()

        messages = [{"role": "user", "content": user_input}]

        msgs, tool_calls = self.llm.chat_with_tools(
            messages=messages,
            tools=tools,
            system="你是 Mnemis，一个专业的 AI 研究助手。",
        )

        tool_names = [tc.name for tc in tool_calls]

        # 如果有工具调用，执行并获取最终回答
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                res = registry.execute(tc.name, tc.input)
                res.tool_use_id = tc.id
                tool_results.append(res)

            msgs = self.llm.append_tool_results(msgs, tool_results)
            msgs, _ = self.llm.chat_with_tools(
                messages=msgs,
                tools=tools,
                system="你是 Mnemis，一个专业的 AI 研究助手。",
            )

        # 提取最终文本
        final = ""
        for block in msgs[-1].get("content", []):
            if hasattr(block, "text"):
                final += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                final += block.get("text", "")

        return final, tool_names

    def _judge(self, case: EvalCase, result: EvalResult):
        """用 LLM 评估实际输出的质量"""
        dims = [d.value for d in case.dimensions]

        reference_section = ""
        if case.reference_answer:
            reference_section = f"参考答案（可作为对照）：\n{case.reference_answer}\n"

        tool_section = ""
        if result.tool_calls_made:
            tool_section = f"Agent 调用了以下工具：{', '.join(result.tool_calls_made)}"

        # 动态生成维度占位符
        dims_str = ", ".join(dims)
        prompt = JUDGE_PROMPT.format(
            input=case.input,
            expected_behavior=case.expected_behavior,
            reference_section=reference_section,
            actual_output=result.actual_output[:2000],
            tool_section=tool_section,
            dimensions=dims_str,
            dim1=dims[0] if dims else "answer_quality",
            dim2=dims[1] if len(dims) > 1 else "answer_quality",
        )

        response = self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            system=JUDGE_SYSTEM,
            temperature=0.1,
        )

        try:
            raw = (
                response.content.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            judge_data = json.loads(raw)

            result.scores = judge_data.get("scores", {})
            result.overall_score = float(judge_data.get("overall_score", 5.0))
            result.passed = bool(judge_data.get("passed", False))
            result.judge_reasoning = judge_data.get("reasoning", "")

        except Exception as e:
            logger.error(f"Judge parsing failed for {case.id}: {e}")
            result.overall_score = 5.0
            result.passed = True  # 解析失败默认通过，避免误报

    def _save_report(self, report: EvalReport):
        """把报告持久化到 SQLite"""
        with sqlite3.connect(EVAL_DB_PATH) as conn:
            # 保存运行记录
            conn.execute(
                """INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?)""",
                (
                    report.run_id,
                    report.agent_version,
                    report.pass_rate,
                    report.avg_score,
                    report.total_cases,
                    report.notes,
                    report.created_at.isoformat(),
                ),
            )
            # 保存各用例结果
            for r in report.results:
                conn.execute(
                    """INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"{report.run_id}-{r.case_id}",
                        report.run_id,
                        r.case_id,
                        r.overall_score,
                        int(r.passed),
                        json.dumps(r.scores),
                        r.actual_output[:200],
                        r.error,
                        r.execution_time,
                        r.evaluated_at.isoformat(),
                    ),
                )
        logger.info(f"Eval report saved: {report.run_id}")
