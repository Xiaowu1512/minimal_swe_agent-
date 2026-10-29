# Path 来自 Python 标准库 pathlib。
# 主要用于寻找项目根目录和 workspace 目录。
from pathlib import Path

# 导入我们阶段 1 实现的 LLM 客户端。
# 当前实际调用 DeepSeek-V3。
from swe_agent.llm.shubiaobiao import ShubiaobiaoLLM

# 导入整个 SWE Agent 的核心类。
# 它负责运行 ReAct 循环。
from swe_agent.agent.agent import SWEAgent

# 导入 PromptBuilder。
# 负责整理任务、工具说明和执行历史。
from swe_agent.agent.prompt import PromptBuilder

# 导入 ActionParser。
# 负责将 LLM 返回的 JSON 转换成 Action。
from swe_agent.parser.action_parser import ActionParser

# 导入工具注册中心。
# 以后 Agent 通过它查找需要执行的工具。
from swe_agent.tools.registry import ToolRegistry


# 导入文件读取、完整写入、文本搜索和局部文本替换工具。
from swe_agent.tools.file_tools import (
    ReadFileTool,       # 读取文件内容。
    WriteFileTool,      # 创建或完整覆盖文件。
    SearchTextTool,     # 搜索文件中的指定文本。
    ReplaceTextTool,    # 只替换指定的文本片段。
)

# 导入刚刚实现的受限终端工具。
# 第一次实验只允许执行 pytest。
from swe_agent.tools.terminal import PytestOnlyTerminalTool


# 定义创建 Agent 的函数。
# 把组装过程独立封装，方便以后扩展。
def create_agent() -> SWEAgent:

    # __file__ 表示当前 main.py 的路径。
    # resolve() 将其转为绝对路径。
    # parent 得到 main.py 所在的项目根目录。
    project_root = Path(__file__).resolve().parent

    # 在项目根目录下找到 workspace。
    # Agent 的文件操作以这个目录为根。
    workspace = project_root / "workspace"

    # 确保 workspace 存在。
    # exist_ok=True 表示目录已经存在时不报错。
    workspace.mkdir(exist_ok=True)

    # 创建工具注册中心。
    registry = ToolRegistry()

    # 注册文件读取工具。
    registry.register(
        ReadFileTool(workspace=workspace)
    )

    # 注册文件写入工具。
    registry.register(
        WriteFileTool(workspace=workspace)
    )

    # 注册文本搜索工具。
    registry.register(
        SearchTextTool(workspace=workspace)
    )
    # 注册局部文本替换工具。
    # LLM 可以通过工具名称 replace_text 调用它。
    registry.register(
        ReplaceTextTool(
            workspace=workspace,  # 限制文件操作的工作目录。
        )
    )

    # 注册受限终端工具。
    # 默认命令最多运行 30 秒。
    registry.register(
        PytestOnlyTerminalTool(
            workspace=workspace,
            timeout=30,
        )
    )

    # 创建 LLM 客户端。
    # API Key 和模型名称从 .env 读取。
    llm = ShubiaobiaoLLM()

    # 创建 JSON 输出解析器。
    parser = ActionParser()

    # 创建 Prompt 构建器。
    prompt_builder = PromptBuilder()

    # 把前面创建的模块注入 SWEAgent。
    # 这就是依赖注入：组件由外部创建并传入。
    agent = SWEAgent(
        llm=llm,
        registry=registry,
        parser=parser,
        prompt_builder=prompt_builder,

        # 最多执行 10 个 Agent 步骤。
        # 避免反复调用 API 导致不必要的费用。
        max_steps=10,
    )

    # 返回组装完成的 Agent。
    return agent


# 定义主函数，作为整个项目的启动入口。
def main() -> None:

    # 创建一个具备真实 LLM 和真实 Tool 的 Agent。
    agent = create_agent()

    # 定义第一次实验的任务。
    # 我们会在下一步创建对应的 calculator 文件。
    # 定义本次用于验证局部文本替换的任务。
    task = (
        "workspace 中有 calculator.py 和 test_calculator.py。"
        "请修复 calculator.py 中的 Bug，使测试通过。"
        "可以使用 read_file、write_file、replace_text、search_text 和 terminal 工具。"
        "如果只需要修改局部代码，请优先使用 replace_text，避免覆盖整个文件。"
        "修改后必须使用 terminal 运行 python -m pytest。"
        "只有确认测试通过后才能输出 finish。"
    )

     # 输出任务说明。
    print("开始执行 SWE Agent 任务：")
    print(task)

    # 运行 Agent。
    # run() 会启动我们之前实现的完整 Agent Loop。
    state = agent.run(task)

    # 输出执行过程。
    print("\n========== Agent 执行历史 ==========")

    # enumerate() 会同时返回序号和历史步骤。
    # start=1 表示序号从 1 开始。
    for index, step in enumerate(state.history, start=1):

        # 输出当前步骤编号。
        print(f"\nStep {index}")

        # 输出 LLM 选择的工具名称。
        print(f"Action: {step.action.tool}")

        # 输出 LLM 给出的简短决策说明。
        print(f"Reasoning: {step.action.reasoning}")

        # 输出工具执行是否成功。
        print(f"Success: {step.observation.success}")

        # 输出工具返回的实际结果。
        print(f"Observation: {step.observation.content}")

    # 输出最终结果。
    print("\n========== 最终结果 ==========")

    # 输出 Agent 的最终回答。
    print(state.final_answer)

    # 输出 Agent 一共执行了多少步。
    print(f"执行步数：{state.step_count}")


# 只有直接运行 main.py 时才执行 main()。
if __name__ == "__main__":

    # 启动主函数。
    main()