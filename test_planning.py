# planning_demo.py
import asyncio
import core.tools
from core.client import LLMClient
from core.planning.planner import GoalPlanner
from core.planning.executor import PlanExecutor


async def demo_goal_planning():
    print("=" * 60)
    print("演示1：目标分解与执行")
    print("=" * 60)

    llm = LLMClient()
    planner = GoalPlanner(llm)
    executor = PlanExecutor(planner, llm, verbose=True)

    # 给一个真实的、有一定复杂度的目标
    goal = (
        "研究并整理 Python 3.12 和 3.13 的核心新特性，"
        "生成一份对比分析报告保存到文件，"
        "并给出从 3.12 升级到 3.13 的建议"
    )

    # 规划
    plan = planner.plan(goal, context={"audience": "Python 后端开发者"})
    print(f"\n规划完成，{len(plan.steps)} 个步骤：")
    for s in plan.steps:
        deps = f" ← {s.depends_on}" if s.depends_on else " ← 无依赖（可立即开始）"
        print(f"  [{s.id}] {s.description[:50]}{deps}")

    # 执行
    print("\n开始执行...")
    plan = await executor.execute(plan)

    print(f"\n{'='*60}")
    print(f"执行完成：{plan.progress_summary()}")
    print(f"重规划次数：{plan.replan_count}")
    print(f"\n最终总结：\n{plan.final_summary}")


async def demo_dynamic_replan():
    print("\n" + "=" * 60)
    print("演示2：动态重规划——步骤失败后自动调整")
    print("=" * 60)

    llm = LLMClient()
    planner = GoalPlanner(llm)

    from core.planning.models import ExecutionPlan, PlanStep, StepStatus

    # 手动构建一个包含「必然失败步骤」的计划
    plan = ExecutionPlan(
        goal="整理 2024 年 AI 领域十大进展并生成报告",
        steps=[
            PlanStep(
                id="step-1",
                description="搜索 2024 年 AI 领域重大事件",
                milestone="获得至少5个有来源的重大事件",
                depends_on=[],
            ),
            PlanStep(
                id="step-2-broken",
                description="从某个不存在的内部数据库获取 AI 论文数据",
                milestone="获得论文数据",
                depends_on=[],
            ),
            PlanStep(
                id="step-3",
                description="综合以上信息写报告",
                milestone="生成 1000 字报告",
                depends_on=["step-1", "step-2-broken"],
            ),
        ],
    )

    print("初始计划：")
    for s in plan.steps:
        print(f"  [{s.id}] {s.description}")

    # 模拟 step-2-broken 失败
    plan.steps[0].status = StepStatus.DONE
    plan.steps[0].result = "找到了 GPT-4o、Claude 3.5、Gemini 2.0 等重大发布..."
    plan.context["step-1"] = plan.steps[0].result

    plan.steps[1].status = StepStatus.FAILED

    # 触发重规划
    print(f"\nstep-2-broken 失败，触发重规划...")
    failure_analysis = "内部数据库不可访问，需要改用网络搜索替代"
    plan = planner.replan(plan, failure_analysis)

    print(f"\n重规划后（共 {len(plan.steps)} 步）：")
    for s in plan.steps:
        status_icon = {
            StepStatus.DONE: "✅",
            StepStatus.FAILED: "❌",
            StepStatus.SKIPPED: "⏭️",
            StepStatus.PENDING: "⏳",
        }.get(s.status, "⏳")
        print(f"  {status_icon} [{s.id}] {s.description[:60]}")

    print(f"\n✅ 动态重规划成功：新增了 {plan.replan_count} 次调整")


def demo_plan_visualization():
    """可视化展示计划的依赖关系"""
    print("\n" + "=" * 60)
    print("演示3：计划依赖关系可视化")
    print("=" * 60)

    llm = LLMClient()
    planner = GoalPlanner(llm)

    plan = planner.plan(
        "开发一个 Python CLI 工具，能搜索 GitHub 仓库并生成分析报告",
        context={"tech_stack": "Python 3.12, Click, Rich"},
    )

    # ASCII 依赖图
    print(f"\n目标：{plan.goal}\n")
    print("依赖关系图：\n")

    # 找出各层（按依赖深度）
    layers: dict[int, list] = {}

    def get_depth(step_id: str, visited: set) -> int:
        if step_id in visited:
            return 0
        visited.add(step_id)
        step = next((s for s in plan.steps if s.id == step_id), None)
        if not step or not step.depends_on:
            return 0
        return 1 + max(get_depth(dep, visited) for dep in step.depends_on)

    for step in plan.steps:
        depth = get_depth(step.id, set())
        layers.setdefault(depth, []).append(step)

    for depth in sorted(layers.keys()):
        steps_in_layer = layers[depth]
        indent = "  " * depth
        parallel = "（并行）" if len(steps_in_layer) > 1 else ""
        print(f"{indent}层 {depth}{parallel}：")
        for s in steps_in_layer:
            effort_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                s.estimated_effort, "⚪"
            )
            print(f"{indent}  {effort_icon} [{s.id}] {s.description[:50]}")
        if depth < max(layers.keys()):
            print(f"{'  ' * (depth+1)}↓")


if __name__ == "__main__":
    asyncio.run(demo_goal_planning())
    asyncio.run(demo_dynamic_replan())
    demo_plan_visualization()
