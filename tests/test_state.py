# 导入刚刚定义的三个核心数据结构。
from swe_agent.agent.state import Action, AgentState, Observation


# pytest 会自动识别以 test_ 开头的函数。
def test_agent_state_records_step():

    # 创建一个模拟用户任务。
    state = AgentState(
        task="修复 calculator 项目"
    )

    # 模拟 LLM 做出一个 Action：
    # 决定读取 calculator.py。
    action = Action(
        tool="read_file",
        arguments={
            "path": "calculator.py"
        },
        reasoning="需要先查看代码",
    )

    # 模拟 read_file 工具返回的结果。
    observation = Observation(
        content="def add(a, b): return a - b",
        success=True,
    )

    # 将这一轮操作记录进 AgentState。
    state.add_step(
        action=action,
        observation=observation,
    )

    # assert 是 Python 的断言语句。
    # 条件不成立时，测试就会失败。
    assert state.step_count == 1

    # 检查 history 中是否真的记录了一步。
    assert len(state.history) == 1

    # 检查记录的工具名称是否正确。
    assert state.history[0].action.tool == "read_file"

    # 检查 Observation 内容是否正确。
    assert state.history[0].observation.success is True


# 测试任务结束功能。
def test_agent_state_finish():

    # 创建 Agent 状态。
    state = AgentState(
        task="修复项目"
    )

    # 模拟 Agent 完成任务。
    state.finish(
        "所有测试已经通过"
    )

    # 检查 finished 是否已经变成 True。
    assert state.finished is True

    # 检查最终答案是否保存正确。
    assert state.final_answer == "所有测试已经通过"