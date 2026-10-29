# json 是 Python 标准库。
#
# 这里主要使用 json.dumps()，
# 把 Python 字典转换成 JSON 字符串，
# 方便把 Action 参数和 metadata 放进 Prompt。
import json

# 导入 AgentState。
#
# PromptBuilder 需要读取 Agent 当前的：
#
# task
# plan
# history
# step_count
from swe_agent.agent.state import AgentState

# 导入 ToolRegistry。
#
# PromptBuilder 需要通过 Registry
# 获取当前 Agent 可以使用的所有工具。
from swe_agent.tools.registry import ToolRegistry


# PromptBuilder 专门负责：
#
# Agent 当前状态
# +
# 可用工具
#
#          ↓
#
# 整理成发送给 LLM 的 Prompt。
class PromptBuilder:

    # build() 是 PromptBuilder 最主要的方法。
    #
    # 输入：
    #
    # state
    #     Agent 当前状态
    #
    # registry
    #     当前已经注册的工具
    #
    # 输出：
    #
    # 一个完整字符串 Prompt。
    def build(
        self,
        state: AgentState,
        registry: ToolRegistry,
    ) -> str:

        # 把当前所有可用工具整理成文本。
        tools_text = self._format_tools(
            registry
        )

        # 把 Agent 之前执行过的历史步骤
        # 整理成文本。
        history_text = self._format_history(
            state
        )

        # 把 Planner 产生的计划整理成文本。
        #
        # 目前 Planner 还没有实现，
        # 因此大多数时候这里会显示“暂无预先计划”。
        plan_text = self._format_plan(
            state
        )

        # 开始构造完整 Prompt。
        #
        # Prompt 会告诉 LLM：
        #
        # 1. 它是谁
        # 2. 当前任务是什么
        # 3. 有哪些工具
        # 4. 已经执行了什么
        # 5. 下一步必须输出什么格式
        prompt = (
            # 定义 LLM 的角色。
            "你是一个 minimal SWE Agent。\n"
            "你的任务是通过读取代码、搜索代码、修改代码和执行终端命令，"
            "逐步完成用户的软件工程任务。\n\n"

            # 告诉模型当前用户任务。
            "========== 用户任务 ==========\n"
            f"{state.task}\n\n"

            # 告诉模型当前计划。
            "========== 当前计划 ==========\n"
            f"{plan_text}\n\n"

            # 告诉模型当前有哪些工具。
            "========== 可用工具 ==========\n"
            f"{tools_text}\n\n"

            # 告诉模型之前发生了什么。
            "========== 执行历史 ==========\n"
            f"{history_text}\n\n"

            # 告诉模型目前已经执行多少步。
            "========== 当前状态 ==========\n"
            f"已经执行 {state.step_count} 步。\n\n"

            # 开始规定 LLM 的行为。
            "========== 下一步要求 ==========\n"

            # 每轮只允许执行一个 Action。
            "你现在只需要决定下一步执行一个操作。\n"

            # 要求模型只能使用我们提供的工具。
            "只能选择上面列出的工具，或者在任务已经完成时选择 finish。\n"

            # reasoning 只要求简短说明。
            #
            # 不需要模型输出冗长推理过程。
            "reasoning 只需要简短说明为什么选择这个操作。\n"

            # 强制要求模型输出 JSON。
            "必须只返回一个 JSON 对象，不要返回其他解释文字。\n\n"

            # 普通工具调用格式。
            "普通工具调用格式：\n"
            "{\n"
            '    "reasoning": "简短说明下一步为什么这样做",\n'
            '    "tool": "工具名称",\n'
            '    "arguments": {\n'
            '        "参数名称": "参数值"\n'
            "    }\n"
            "}\n\n"

            # finish 是一个特殊 Action。
            #
            # 它不是真正的 Tool，
            # 而是用来告诉 Agent Loop：
            #
            # 当前任务已经完成，可以停止循环。
            "如果任务已经完成，使用下面的格式：\n"
            "{\n"
            '    "reasoning": "任务已经完成",\n'
            '    "tool": "finish",\n'
            '    "arguments": {\n'
            '        "answer": "向用户说明最终完成情况"\n'
            "    }\n"
            "}\n"
        )

        # 返回最终构建完成的 Prompt。
        return prompt

    # _format_tools() 是内部辅助方法。
    #
    # "_" 开头表示：
    # 这个方法主要供 PromptBuilder 自己使用。
    def _format_tools(
        self,
        registry: ToolRegistry,
    ) -> str:

        # registry.all()
        # 会返回所有已经注册的 Tool 对象。
        tools = registry.all()

        # 如果现在一个工具都没有，
        # 返回明确提示。
        if not tools:
            return "当前没有可用工具。"

        # 创建一个空列表，
        # 用来保存每个工具的说明。
        tool_lines: list[str] = []

        # 遍历所有 Tool。
        for tool in tools:

            # 每个 Tool 都有：
            #
            # name
            # description
            #
            # 这是 BaseTool 强制规定的统一接口。
            tool_line = (
                f"- {tool.name}: "
                f"{tool.description}"
            )

            # 把当前工具说明加入列表。
            tool_lines.append(
                tool_line
            )

        # 使用换行符把多个工具连接起来。
        #
        # 最终可能得到：
        #
        # - read_file: ...
        # - write_file: ...
        # - search_text: ...
        # - terminal: ...
        return "\n".join(
            tool_lines
        )

    # _format_plan() 用来整理任务计划。
    def _format_plan(
        self,
        state: AgentState,
    ) -> str:

        # 如果 plan 是空列表，
        # 说明当前还没有 Planner 产生计划。
        if not state.plan:
            return "暂无预先计划。"

        # 创建列表保存格式化后的计划。
        plan_lines: list[str] = []

        # enumerate() 可以同时获得：
        #
        # 序号
        # 计划内容
        #
        # start=1 表示编号从 1 开始。
        for index, item in enumerate(
            state.plan,
            start=1,
        ):

            # 例如：
            #
            # 1. 查看项目结构
            # 2. 运行测试
            plan_lines.append(
                f"{index}. {item}"
            )

        # 把所有计划连接成一段文本。
        return "\n".join(
            plan_lines
        )

    # _format_history() 用来把 Agent 已执行过的
    # Action / Observation 转换成人类和 LLM 都容易阅读的文本。
    def _format_history(
        self,
        state: AgentState,
    ) -> str:

        # 如果 Agent 还没有执行任何操作。
        if not state.history:
            return "暂无执行历史。"

        # 保存每一个历史步骤。
        history_parts: list[str] = []

        # 遍历 AgentState 中保存的所有 AgentStep。
        for index, step in enumerate(
            state.history,
            start=1,
        ):

            # json.dumps()：
            #
            # Python 字典
            #      ↓
            # JSON 字符串
            #
            # ensure_ascii=False
            # 可以让中文直接显示，
            # 而不是转换成 \uXXXX。
            arguments_text = json.dumps(
                step.action.arguments,
                ensure_ascii=False,
            )

            # metadata 同样转换成 JSON。
            metadata_text = json.dumps(
                step.observation.metadata,
                ensure_ascii=False,
            )

            # 整理当前这一轮 AgentStep。
            step_text = (
                f"Step {index}\n"
                f"Reasoning: {step.action.reasoning}\n"
                f"Action: {step.action.tool}\n"
                f"Arguments: {arguments_text}\n"
                f"Observation success: {step.observation.success}\n"
                f"Observation:\n{step.observation.content}\n"
                f"Metadata: {metadata_text}"
            )

            # 保存当前步骤。
            history_parts.append(
                step_text
            )

        # 使用两个换行符分隔不同 AgentStep，
        # 让 Prompt 更容易阅读。
        return "\n\n".join(
            history_parts
        )