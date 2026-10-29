# Path 这里其实不需要使用，所以不导入。
# 本测试主要验证 PromptBuilder 是否能正确整合：
#
# 1. 用户任务
# 2. ToolRegistry 中的工具
# 3. Agent 的 plan
# 4. Agent 的 history
# 5. finish 结束格式

# 导入我们已经定义好的 Agent 数据结构。
from swe_agent.agent.state import (
    Action,
    AgentState,
    Observation,
)

# 导入这一步需要测试的 PromptBuilder。
from swe_agent.agent.prompt import PromptBuilder

# 导入 ToolRegistry。
#
# PromptBuilder 会从 ToolRegistry 中读取当前可用工具。
from swe_agent.tools.registry import ToolRegistry

# 导入 BaseTool。
#
# 为了测试 PromptBuilder，
# 我们会创建一个假的 Tool 放进 Registry。
from swe_agent.tools.base import BaseTool


# FakeTool 是专门用于测试的假工具。
#
# 我们不需要真的读取文件或执行命令，
# 只需要验证 PromptBuilder 是否能读取：
#
# name
# description
class FakeTool(BaseTool):

    # 返回工具名称。
    @property
    def name(self) -> str:

        # 给这个测试工具固定名称。
        return "fake_tool"

    # 返回工具说明。
    @property
    def description(self) -> str:

        # PromptBuilder 应该把这段内容放进 Prompt。
        return "这是一个用于测试的工具"

    # BaseTool 要求所有子类都必须实现 run()。
    #
    # 这个测试里实际上不会调用 run()，
    # 但为了满足 BaseTool 的接口要求，
    # 仍然需要实现。
    def run(
        self,
        arguments: dict,
    ) -> Observation:

        # 返回一个简单的 Observation。
        return Observation(
            content="fake tool executed",
            success=True,
        )


# 测试 1：
# 检查 Prompt 中是否包含用户任务。
def test_prompt_contains_task():

    # 创建 AgentState。
    #
    # task 就是用户交给 Agent 的任务。
    state = AgentState(
        task="修复 calculator 项目"
    )

    # 创建一个空的 ToolRegistry。
    registry = ToolRegistry()

    # 创建 PromptBuilder。
    builder = PromptBuilder()

    # 构建 Prompt。
    prompt = builder.build(
        state=state,
        registry=registry,
    )

    # Prompt 中应该包含用户任务。
    assert "修复 calculator 项目" in prompt

    # Prompt 中应该包含“用户任务”这一部分。
    assert "用户任务" in prompt


# 测试 2：
# 检查 PromptBuilder 是否能把
# ToolRegistry 中的工具写入 Prompt。
def test_prompt_contains_tools():

    # 创建 AgentState。
    state = AgentState(
        task="测试工具"
    )

    # 创建 ToolRegistry。
    registry = ToolRegistry()

    # 创建假的 Tool。
    fake_tool = FakeTool()

    # 把工具注册进去。
    registry.register(
        fake_tool
    )

    # 创建 PromptBuilder。
    builder = PromptBuilder()

    # 构建 Prompt。
    prompt = builder.build(
        state=state,
        registry=registry,
    )

    # Prompt 中应该出现工具名称。
    assert "fake_tool" in prompt

    # Prompt 中应该出现工具说明。
    assert "这是一个用于测试的工具" in prompt


# 测试 3：
# 检查 AgentState 中的 plan
# 是否会被正确加入 Prompt。
def test_prompt_contains_plan():

    # 创建 AgentState。
    state = AgentState(
        task="修复项目"
    )

    # 手动给 Agent 添加一个简单计划。
    state.plan = [
        "运行测试",
        "读取错误代码",
        "修改代码",
    ]

    # 创建空的 ToolRegistry。
    registry = ToolRegistry()

    # 创建 PromptBuilder。
    builder = PromptBuilder()

    # 构建 Prompt。
    prompt = builder.build(
        state=state,
        registry=registry,
    )

    # Prompt 中应该出现第一步计划。
    assert "1. 运行测试" in prompt

    # Prompt 中应该出现第二步计划。
    assert "2. 读取错误代码" in prompt

    # Prompt 中应该出现第三步计划。
    assert "3. 修改代码" in prompt


# 测试 4：
# 检查已经执行过的 Action 和 Observation
# 是否会被写入 Prompt 的执行历史中。
def test_prompt_contains_history():

    # 创建 AgentState。
    state = AgentState(
        task="修复 calculator"
    )

    # 创建一个模拟 Action。
    action = Action(

        # 模拟 Agent 决定读取文件。
        tool="read_file",

        # 模拟工具参数。
        arguments={
            "path": "calculator.py"
        },

        # 模拟模型给出的简短原因。
        reasoning="需要先查看代码",
    )

    # 创建对应的 Observation。
    observation = Observation(

        # 模拟读取到的文件内容。
        content="def add(a, b): return a - b",

        # 工具正常执行。
        success=True,

        # 保存额外信息。
        metadata={
            "path": "calculator.py"
        },
    )

    # 把这一轮 Action + Observation
    # 加入 AgentState 的 history。
    state.add_step(
        action=action,
        observation=observation,
    )

    # 创建空的 ToolRegistry。
    registry = ToolRegistry()

    # 创建 PromptBuilder。
    builder = PromptBuilder()

    # 构建 Prompt。
    prompt = builder.build(
        state=state,
        registry=registry,
    )

    # Prompt 应该包含 Agent 的 reasoning。
    assert "需要先查看代码" in prompt

    # Prompt 应该包含工具名称。
    assert "read_file" in prompt

    # Prompt 应该包含文件路径。
    assert "calculator.py" in prompt

    # Prompt 应该包含工具返回的文件内容。
    assert "def add(a, b): return a - b" in prompt

    # state.add_step() 后，
    # step_count 应该已经变成 1。
    #
    # Prompt 中应该把这个状态告诉 LLM。
    assert "已经执行 1 步" in prompt


# 测试 5：
# 检查 Prompt 是否明确告诉 LLM：
#
# 任务完成时应该使用 finish。
def test_prompt_contains_finish_instruction():

    # 创建 AgentState。
    state = AgentState(
        task="完成任务"
    )

    # 创建空的 ToolRegistry。
    registry = ToolRegistry()

    # 创建 PromptBuilder。
    builder = PromptBuilder()

    # 构建 Prompt。
    prompt = builder.build(
        state=state,
        registry=registry,
    )

    # Prompt 中必须包含 finish。
    #
    # 后面的 Agent Loop 会根据：
    #
    # action.tool == "finish"
    #
    # 判断是否结束循环。
    assert '"tool": "finish"' in prompt

    # finish 的 arguments 中
    # 还应该包含最终回答 answer。
    assert '"answer"' in prompt