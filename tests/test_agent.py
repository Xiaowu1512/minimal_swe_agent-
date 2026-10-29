# 导入 BaseLLM。
#
# FakeLLM 会继承 BaseLLM，
# 从而模拟我们真实使用的 DeepSeek-V3。
from swe_agent.llm.base import BaseLLM

# 导入 BaseTool。
#
# FakeTool 会继承 BaseTool，
# 用来模拟真实的 read_file、terminal 等工具。
from swe_agent.tools.base import BaseTool

# 导入 ToolRegistry。
#
# SWEAgent 会通过 ToolRegistry 查找工具。
from swe_agent.tools.registry import ToolRegistry

# 导入 Observation。
#
# FakeTool 执行以后同样需要返回标准 Observation。
from swe_agent.agent.state import Observation

# 导入 ActionParser。
#
# Agent Loop 会使用它把 LLM 返回的 JSON
# 转换成 Action。
from swe_agent.parser.action_parser import ActionParser

# 导入 PromptBuilder。
#
# Agent Loop 每轮都会通过它生成 Prompt。
from swe_agent.agent.prompt import PromptBuilder

# 导入我们现在真正要测试的 SWEAgent。
from swe_agent.agent.agent import SWEAgent


# FakeLLM 是一个假的大语言模型。
#
# 它不会调用任何 API，
# 因此：
#
# 1. 不消耗 API 费用
# 2. 输出完全可控
# 3. 非常适合测试 Agent Loop
class FakeLLM(BaseLLM):

    # 初始化时提前给它准备一组固定回答。
    def __init__(
        self,
        responses: list[str],
    ):

        # 保存所有预设回答。
        self.responses = responses

        # index 表示下一次应该返回哪一个回答。
        self.index = 0

    # 实现 BaseLLM 要求的 generate()。
    def generate(
        self,
        prompt: str,
    ) -> str:

        # 如果还有没有使用的回答，
        # 就按顺序取出。
        if self.index < len(self.responses):

            # 取得当前回答。
            response = self.responses[self.index]

            # index 加 1，
            # 下一次 generate() 就会返回下一条。
            self.index += 1

            # 返回模拟的 LLM 输出。
            return response

        # 如果预设回答已经全部用完，
        # 直接抛出异常。
        #
        # 正常测试中不应该走到这里，
        # 如果出现说明 Agent 多调用了 LLM。
        raise RuntimeError(
            "FakeLLM 没有更多预设回答"
        )


# FakeTool 是一个简单的假工具。
#
# 用它来确认：
#
# Agent 是否真的通过 ToolRegistry
# 找到了 Tool 并执行 run()。
class FakeTool(BaseTool):

    # 定义工具名称。
    @property
    def name(self) -> str:

        # 使用 fake_tool 作为测试工具名。
        return "fake_tool"

    # 定义工具说明。
    @property
    def description(self) -> str:

        # PromptBuilder 会把这个说明放进 Prompt。
        return "这是一个用于测试 Agent Loop 的工具"

    # 实现真正的工具执行方法。
    def run(
        self,
        arguments: dict,
    ) -> Observation:

        # 返回一个固定 Observation。
        #
        # 同时把传入参数放到 metadata，
        # 方便测试 Agent 有没有正确传递 arguments。
        return Observation(
            content="fake tool executed",
            success=True,
            metadata={
                "arguments": arguments,
            },
        )


# CrashTool 用来模拟一个内部发生异常的工具。
#
# 它主要测试：
#
# 即使某个 Tool 自己崩了，
# SWEAgent 也不应该整个程序直接退出。
class CrashTool(BaseTool):

    # 工具名称。
    @property
    def name(self) -> str:
        return "crash_tool"

    # 工具说明。
    @property
    def description(self) -> str:
        return "这个工具会故意抛出异常"

    # run() 被调用时直接抛出异常。
    def run(
        self,
        arguments: dict,
    ) -> Observation:

        # RuntimeError 是 Python 常见运行时异常。
        raise RuntimeError(
            "模拟工具内部错误"
        )


# 创建一个辅助函数，
# 专门负责组装测试需要的 SWEAgent。
#
# 这样每个测试就不用重复写：
#
# registry
# parser
# prompt_builder
# agent
def create_test_agent(
    responses: list[str],
    max_steps: int = 20,
    tools: list[BaseTool] | None = None,
) -> SWEAgent:

    # 创建假的 LLM。
    llm = FakeLLM(
        responses=responses
    )

    # 创建 ToolRegistry。
    registry = ToolRegistry()

    # 如果测试传入了 Tool，
    # 就逐个注册进去。
    if tools is not None:

        # 遍历所有工具。
        for tool in tools:

            # 注册工具。
            registry.register(
                tool
            )

    # 创建 ActionParser。
    parser = ActionParser()

    # 创建 PromptBuilder。
    prompt_builder = PromptBuilder()

    # 创建真正的 SWEAgent。
    agent = SWEAgent(
        llm=llm,
        registry=registry,
        parser=parser,
        prompt_builder=prompt_builder,
        max_steps=max_steps,
    )

    # 返回组装完成的 Agent。
    return agent


# 测试 1：
# 验证最基本的 Agent Loop：
#
# LLM
# ↓
# Tool
# ↓
# Observation
# ↓
# 下一轮
# ↓
# finish
def test_agent_executes_tool_and_finishes():

    # 第一次 LLM 输出：
    # 调用 fake_tool。
    first_response = (
        '{"reasoning": "先执行测试工具", '
        '"tool": "fake_tool", '
        '"arguments": {"value": 123}}'
    )

    # 第二次 LLM 输出：
    # 任务结束。
    second_response = (
        '{"reasoning": "任务已经完成", '
        '"tool": "finish", '
        '"arguments": {"answer": "测试任务完成"}}'
    )

    # 创建 Agent，
    # 并注册 FakeTool。
    agent = create_test_agent(
        responses=[
            first_response,
            second_response,
        ],
        tools=[
            FakeTool(),
        ],
    )

    # 正式运行 Agent。
    state = agent.run(
        "执行一个测试任务"
    )

    # Agent 最终应该已经结束。
    assert state.finished is True

    # 最终答案应该来自 finish Action。
    assert state.final_answer == "测试任务完成"

    # 一共执行两步：
    #
    # Step 1：fake_tool
    # Step 2：finish
    assert state.step_count == 2

    # history 中也应该有两条记录。
    assert len(state.history) == 2

    # 第一轮应该执行 fake_tool。
    assert state.history[0].action.tool == "fake_tool"

    # FakeTool 返回的 Observation
    # 应该被正确记录。
    assert state.history[0].observation.content == "fake tool executed"

    # Action arguments 应该成功传给 Tool。
    assert (
        state.history[0]
        .observation
        .metadata["arguments"]["value"]
        == 123
    )

    # 第二轮应该是 finish。
    assert state.history[1].action.tool == "finish"


# 测试 2：
# 验证 LLM 选择不存在的 Tool 时，
# Agent 不会崩溃，
# 而是把错误记录到 Observation。
def test_agent_handles_unknown_tool():

    # 第一次故意调用一个不存在的工具。
    first_response = (
        '{"reasoning": "尝试调用不存在的工具", '
        '"tool": "not_exist", '
        '"arguments": {}}'
    )

    # 第二轮正常 finish。
    second_response = (
        '{"reasoning": "结束测试", '
        '"tool": "finish", '
        '"arguments": {"answer": "结束"}}'
    )

    # 创建没有注册任何工具的 Agent。
    agent = create_test_agent(
        responses=[
            first_response,
            second_response,
        ]
    )

    # 运行 Agent。
    state = agent.run(
        "测试未知工具"
    )

    # Agent 最终仍然应该正常结束。
    assert state.finished is True

    # 第一轮 Observation 应该表示失败。
    assert state.history[0].observation.success is False

    # 返回内容应该告诉 Agent：
    # 工具不存在。
    assert "未知工具" in state.history[0].observation.content

    # Agent 应该继续执行第二轮，
    # 而不是第一轮直接崩溃。
    assert state.step_count == 2


# 测试 3：
# 验证 LLM 第一次返回非法 JSON 时，
# Agent 是否能够记录解析失败，
# 然后继续下一轮。
def test_agent_handles_invalid_llm_output():

    # 第一轮故意返回一个不是 JSON 的字符串。
    first_response = (
        "这不是合法的 JSON"
    )

    # 第二轮返回合法 finish。
    second_response = (
        '{"reasoning": "修正输出格式并结束", '
        '"tool": "finish", '
        '"arguments": {"answer": "已结束"}}'
    )

    # 创建测试 Agent。
    agent = create_test_agent(
        responses=[
            first_response,
            second_response,
        ]
    )

    # 运行 Agent。
    state = agent.run(
        "测试错误 JSON"
    )

    # Agent 最终应该能够正常结束。
    assert state.finished is True

    # 第一轮应该被记录成特殊 Action：
    # invalid_output。
    assert state.history[0].action.tool == "invalid_output"

    # 第一轮 Observation 应该失败。
    assert state.history[0].observation.success is False

    # Observation 应该包含解析失败信息。
    assert "Action 解析失败" in state.history[0].observation.content

    # 原始错误输出应该保存在 metadata 中，
    # 方便以后调试。
    assert (
        state.history[0]
        .observation
        .metadata["raw_output"]
        == first_response
    )

    # 第二轮 finish 后，
    # 一共应该执行两步。
    assert state.step_count == 2


# 测试 4：
# 验证 Tool 自己抛出未处理异常时，
# SWEAgent 是否能够捕获异常，
# 而不是整个程序直接退出。
def test_agent_handles_tool_exception():

    # 第一轮调用故意会崩溃的 crash_tool。
    first_response = (
        '{"reasoning": "执行异常工具", '
        '"tool": "crash_tool", '
        '"arguments": {}}'
    )

    # 第二轮正常结束。
    second_response = (
        '{"reasoning": "结束测试", '
        '"tool": "finish", '
        '"arguments": {"answer": "完成"}}'
    )

    # 创建 Agent，
    # 注册 CrashTool。
    agent = create_test_agent(
        responses=[
            first_response,
            second_response,
        ],
        tools=[
            CrashTool(),
        ],
    )

    # 运行 Agent。
    state = agent.run(
        "测试工具异常"
    )

    # Agent 应该没有因为 Tool 异常而崩溃。
    assert state.finished is True

    # 第一轮 Observation 应该失败。
    assert state.history[0].observation.success is False

    # 错误信息中应该说明：
    # 工具出现了未处理异常。
    assert "工具执行出现未处理异常" in (
        state.history[0].observation.content
    )

    # 原始异常信息也应该被保留下来。
    assert "模拟工具内部错误" in (
        state.history[0].observation.content
    )


# 测试 5：
# 验证 max_steps 是否真正能够阻止
# Agent 无限循环。
def test_agent_stops_at_max_steps():

    # 准备两次完全正常的 Tool 调用，
    # 但故意不给 finish。
    first_response = (
        '{"reasoning": "执行第一次工具调用", '
        '"tool": "fake_tool", '
        '"arguments": {}}'
    )

    second_response = (
        '{"reasoning": "执行第二次工具调用", '
        '"tool": "fake_tool", '
        '"arguments": {}}'
    )

    # max_steps 设置成 2。
    #
    # Agent 执行两步以后，
    # 应该被强制停止。
    agent = create_test_agent(
        responses=[
            first_response,
            second_response,
        ],
        max_steps=2,
        tools=[
            FakeTool(),
        ],
    )

    # 运行 Agent。
    state = agent.run(
        "测试最大执行步数"
    )

    # 即使 LLM 没有主动输出 finish，
    # Agent 最终也应该被标记为 finished。
    assert state.finished is True

    # 实际执行步数应该正好等于 2。
    assert state.step_count == 2

    # 最终答案应该说明达到了最大步数。
    assert "达到最大执行步数" in state.final_answer

    # history 中应该只有两个真正执行过的 Action。
    assert len(state.history) == 2