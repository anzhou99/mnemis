# test_communication.py
from core.multi_agent.models import (
    AgentTask, AgentRole, TaskStatus, ResearchPlan
)
from core.multi_agent.context import SharedContext
from core.multi_agent.queue import TaskQueue


def test_basic_flow():
    """验证任务队列和共享上下文的基本工作流"""
    print("=== 测试基础通信流 ===\n")

    # 1. 创建研究计划（手动定义任务，不用 LLM）
    t_research = AgentTask(
        id="r1",
        role=AgentRole.RESEARCHER,
        instruction="研究 Python 3.13 的新特性",
    )
    t_write = AgentTask(
        id="w1",
        role=AgentRole.WRITER,
        instruction="根据研究结果写文章",
        depends_on=["r1"],   # 依赖研究任务
    )
    t_review = AgentTask(
        id="rev1",
        role=AgentRole.REVIEWER,
        instruction="审核文章质量",
        depends_on=["w1"],   # 依赖写作任务
    )

    plan = ResearchPlan(topic="Python 3.13", tasks=[t_research, t_write, t_review])
    context = SharedContext(topic="Python 3.13")
    queue = TaskQueue(plan)

    # 2. 第一轮：只有 research 任务 ready（write 和 review 有依赖）
    ready = queue.get_ready_tasks()
    assert len(ready) == 1 and ready[0].id == "r1"
    print(f"✓ 第1轮 ready 任务：{[t.id for t in ready]}")

    # 3. 执行研究任务
    queue.mark_running("r1")
    fake_research = "Python 3.13 引入了实验性 JIT 编译器，性能提升约 20%"
    context.set_result("r1", fake_research, "researcher")
    queue.mark_done("r1", fake_research)

    # 4. 第二轮：write 任务 ready（依赖 r1 已完成）
    ready = queue.get_ready_tasks()
    assert len(ready) == 1 and ready[0].id == "w1"
    print(f"✓ 第2轮 ready 任务：{[t.id for t in ready]}")

    # 5. 执行写作任务
    queue.mark_running("w1")
    # Writer 从 context 读取研究结果
    research_data = context.get_result("r1")
    assert research_data == fake_research
    print(f"✓ Writer 读到研究结果：{research_data[:40]}...")

    fake_draft = "# Python 3.13 深度解析\n\n本文分析了 Python 3.13 的核心新特性..."
    context.set_result("w1", fake_draft, "writer")
    queue.mark_done("w1", fake_draft)

    # 6. 第三轮：review 任务 ready
    ready = queue.get_ready_tasks()
    assert len(ready) == 1 and ready[0].id == "rev1"
    print(f"✓ 第3轮 ready 任务：{[t.id for t in ready]}")

    queue.mark_running("rev1")
    fake_review = "文章结构清晰，建议增加代码示例并补充性能测试数据。"
    context.set_result("rev1", fake_review, "reviewer")
    queue.mark_done("rev1", fake_review)

    # 7. 所有任务完成
    assert queue.is_all_done()
    print(f"✓ 所有任务完成")
    print(f"\n{queue.status_report()}")
    print(f"\n上下文快照：{context.snapshot()}")


def test_cascade_skip():
    """验证任务失败时级联跳过逻辑"""
    print("\n=== 测试级联跳过 ===\n")

    t1 = AgentTask(id="r1", role=AgentRole.RESEARCHER, instruction="研究")
    t2 = AgentTask(id="w1", role=AgentRole.WRITER, instruction="写作", depends_on=["r1"])
    t3 = AgentTask(id="rev1", role=AgentRole.REVIEWER, instruction="审核", depends_on=["w1"])

    plan = ResearchPlan(topic="测试", tasks=[t1, t2, t3])
    queue = TaskQueue(plan)

    # 研究任务失败
    queue.mark_running("r1")
    queue.mark_failed("r1", "搜索 API 超时")

    # 验证级联跳过
    assert t2.status == TaskStatus.SKIPPED
    assert t3.status == TaskStatus.SKIPPED
    assert queue.is_all_done()
    print(f"✓ 研究失败后，写作和审核均被跳过")
    print(f"\n{queue.status_report()}")


if __name__ == "__main__":
    test_basic_flow()
    test_cascade_skip()
    print("\n✅ 所有通信测试通过")