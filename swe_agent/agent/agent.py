# 导入 BaseLLM。
#
# BaseLLM 是我们之前定义的大语言模型统一接口。
# Agent 只依赖 BaseLLM，而不直接依赖具体的 DeepSeek 或其他模型。
from swe_agent.llm.base import BaseLLM

# 导入 AgentState、Action 和 Observation。
#
# AgentState：
# 保存整个 Agent 当前的运行状态。
#
# Action：
# 表示 LLM 决定下一步做什么。
#
# Observation：
# 表示执行 Action 后得到的结果。
from swe_agent.agent.state import (
    Action,
    AgentState,
    Observation,
)

# 导入 PromptBuilder。
#
# 它负责把：
#
# 用户任务
# 当前计划
# 历史执行记录
# 可用工具
#
# 整理成发送给 LLM 的 Prompt。
from swe_agent.agent.prompt import PromptBuilder

# 导入 ActionParser。
#
# 它负责把 LLM 返回的 JSON 字符串
# 转换成标准 Action 对象。
from swe_agent.parser.action_parser import ActionParser

# 导入 ToolRegistry。
#
# Agent 会通过 ToolRegistry
# 根据 Action.tool 找到真正要执行的 Tool。
from swe_agent.tools.registry import ToolRegistry


# SWEAgent 是整个 minimal SWE Agent 的核心类。
#
# 它负责把前面已经实现的模块真正串起来：
#
# Prompt
#   ↓
# LLM
#   ↓
# Parser
#   ↓
# Tool
#   ↓
# Observation
#   ↓
# State
#   ↓
# 下一轮循环
class SWEAgent:

    # __init__() 用来接收 Agent 运行所需要的各个模块。
    #
    # 我们采用“依赖注入”的方式：
    #
    # Agent 不自己创建 LLM、Registry 等对象，
    # 而是从外部传进来。
    #
    # 这样以后更换模型、Parser 或工具系统时，
    # 不需要修改 SWEAgent 本身。
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        parser: ActionParser,
        prompt_builder: PromptBuilder,
        max_steps: int = 20,
    ):

        # 保存 LLM。
        #
        # 当前实际传入的会是 ShubiaobiaoLLM，
        # 也就是我们目前使用的 DeepSeek-V3。
        self.llm = llm

        # 保存 ToolRegistry。
        #
        # 里面以后会注册：
        #
        # read_file
        # write_file
        # search_text
        # terminal
        self.registry = registry

        # 保存 ActionParser。
        self.parser = parser

        # 保存 PromptBuilder。
        self.prompt_builder = prompt_builder

        # max_steps 用来限制 Agent 最多执行多少步。
        #
        # 防止：
        #
        # LLM 一直重复操作
        # 工具不断失败
        # Agent 陷入无限循环
        self.max_steps = max_steps

    # run() 是整个 Agent 的主入口。
    #
    # 用户给出一个 task，
    # Agent 就从这里开始运行。
    def run(
        self,
        task: str,
    ) -> AgentState:

        # 根据用户任务创建一个新的 AgentState。
        #
        # 例如：
        #
        # task =
        # "修复 calculator 项目，使所有测试通过"
        state = AgentState(
            task=task
        )

        # while 是整个 Agent 最核心的循环。
        #
        # 只要：
        #
        # 任务还没有 finished
        #
        # 并且：
        #
        # 执行步数没有超过 max_steps
        #
        # Agent 就会继续运行。
        while (
            not state.finished
            and state.step_count < self.max_steps
        ):

            # 第一步：
            #
            # 根据当前 AgentState 和 ToolRegistry
            # 构建这一轮要发送给 LLM 的 Prompt。
            prompt = self.prompt_builder.build(
                state=state,
                registry=self.registry,
            )

            # 第二步：
            #
            # 调用 LLM。
            #
            # 当前实际执行的是：
            #
            # DeepSeek-V3
            #
            # LLM 应该返回一个 JSON 字符串。
            # 打印当前 Agent 运行到第几轮。
            print(f"\n[Agent] 第 {state.step_count + 1} 轮开始", flush=True)

            # 告诉我们程序即将调用 LLM。
            print("[Agent] 正在请求 DeepSeek-V3...", flush=True)

            # 真正调用大语言模型 API。
            llm_output = self.llm.generate(prompt)

            # API 返回后，立即打印模型输出。
            print(f"[Agent] LLM 返回：{llm_output}", flush=True)
            

            # 第三步：
            #
            # 尝试把 LLM 输出解析成 Action。
            try:

                # parse()：
                #
                # JSON 字符串
                #      ↓
                # Action
                action = self.parser.parse(
                    llm_output
                )

            # 如果 LLM 没有按照要求返回合法 JSON，
            # ActionParser 会抛出 ValueError。
            except ValueError as error:

                # 这里创建一个特殊的 Action。
                #
                # invalid_output 并不是真正的 Tool。
                #
                # 它只是为了把这次“LLM 输出格式错误”
                # 记录到 AgentState.history 中。
                action = Action(
                    tool="invalid_output",
                    arguments={},
                    reasoning="LLM 输出无法解析",
                )

                # 创建失败的 Observation。
                #
                # 下一轮 PromptBuilder 会把这条错误信息
                # 放进历史记录重新发送给 LLM。
                #
                # 这样 LLM 有机会自行修正输出格式。
                observation = Observation(
                    content=(
                        f"Action 解析失败：{str(error)}。"
                        "下一轮必须严格返回合法 JSON。"
                    ),
                    success=False,

                    # 保存模型原始输出，
                    # 方便以后调试 Agent。
                    metadata={
                        "raw_output": llm_output,
                    },
                )

                # 把本轮失败记录进 AgentState。
                state.add_step(
                    action=action,
                    observation=observation,
                )

                # continue 表示：
                #
                # 不再执行这一轮后面的代码，
                # 直接进入下一轮 while 循环。
                continue

            # 第四步：
            #
            # 检查模型是不是决定结束任务。
            #
            # finish 不是 ToolRegistry 中真正注册的工具，
            # 而是 Agent Loop 自己处理的特殊 Action。
            if action.tool == "finish":

                # 从 arguments 中取得最终回答。
                #
                # 正常格式：
                #
                # {
                #     "tool": "finish",
                #     "arguments": {
                #         "answer": "任务已完成"
                #     }
                # }
                final_answer = action.arguments.get(
                    "answer",
                    "",
                )

                # answer 正常情况下应该是字符串。
                #
                # 如果模型意外返回其他类型，
                # 就统一转换成字符串，
                # 防止 Agent 因类型问题直接崩溃。
                if not isinstance(
                    final_answer,
                    str,
                ):
                    final_answer = str(
                        final_answer
                    )

                # 如果模型没有提供 answer，
                # 给一个默认结束信息。
                if not final_answer:
                    final_answer = "任务已结束。"

                # 创建 finish 对应的 Observation。
                observation = Observation(
                    content=final_answer,
                    success=True,
                    metadata={
                        "finished": True,
                    },
                )

                # 把 finish 也记录到历史中。
                #
                # 这样以后查看完整 trajectory 时，
                # 能知道 Agent 最终是在哪一步结束的。
                state.add_step(
                    action=action,
                    observation=observation,
                )

                # 调用 AgentState.finish()。
                #
                # 它会：
                #
                # state.finished = True
                #
                # 并保存 final_answer。
                state.finish(
                    final_answer
                )

                # break 表示：
                #
                # 立即退出 while 循环。
                break

            # 第五步：
            #
            # 如果不是 finish，
            # 就说明模型想调用一个真正的 Tool。
            try:

                # 根据 action.tool
                # 从 ToolRegistry 中查找工具。
                #
                # 例如：
                #
                # action.tool = "read_file"
                #
                # 得到：
                #
                # ReadFileTool 对象
                tool = self.registry.get(
                    action.tool
                )

            # 如果 LLM 选择了一个不存在的工具，
            # ToolRegistry.get() 会抛出 KeyError。
            except KeyError:

                # 创建失败 Observation。
                observation = Observation(
                    content=(
                        f"未知工具：{action.tool}。"
                        "请选择已经注册的工具。"
                    ),
                    success=False,

                    # 把当前真正可用的工具名称
                    # 告诉下一轮 LLM。
                    metadata={
                        "available_tools": self.registry.names(),
                    },
                )

                # 把错误记录下来。
                state.add_step(
                    action=action,
                    observation=observation,
                )

                # 进入下一轮。
                continue

            # 第六步：
            #
            # 真正执行 Tool。
            try:

                # 把 Action 中的 arguments
                # 直接交给 Tool。
                #
                # 例如：
                #
                # action.arguments =
                # {
                #     "path": "calculator.py"
                # }
                #
                # 最终：
                #
                # ReadFileTool.run(...)
                observation = tool.run(
                    action.arguments
                )

            # 正常情况下，
            # 我们之前写的 Tool 自己已经会处理大多数异常。
            #
            # 这里再增加一层保护，
            # 防止某个 Tool 出现未处理异常，
            # 导致整个 Agent 程序直接退出。
            except Exception as error:

                # 把异常转换成失败 Observation。
                observation = Observation(
                    content=(
                        "工具执行出现未处理异常："
                        f"{str(error)}"
                    ),
                    success=False,
                )

            # 第七步：
            #
            # 把本轮：
            #
            # Action
            # +
            # Observation
            #
            # 保存进 AgentState。
            state.add_step(
                action=action,
                observation=observation,
            )

            # 到这里，
            # while 会自动回到最开始。
            #
            # 下一轮 PromptBuilder 会看到刚刚新增的 history，
            # 然后 LLM 根据新的 Observation
            # 决定下一步做什么。

        # 如果退出 while 后，
        # Agent 仍然没有主动 finish，
        #
        # 最常见原因就是：
        #
        # step_count 已经达到 max_steps。
        if not state.finished:

            # 强制结束 Agent，
            # 防止无限运行。
            state.finish(
                (
                    f"达到最大执行步数 "
                    f"{self.max_steps}，任务停止。"
                )
            )

        # 返回完整 AgentState。
        #
        # 而不是只返回 final_answer。
        #
        # 这样调用方以后不仅可以看到最终结果，
        # 还可以查看：
        #
        # history
        # step_count
        # plan
        # final_answer
        #
        # 对调试 SWE Agent 非常有帮助。
        return state