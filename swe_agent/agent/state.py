# dataclass 是 Python 标准库提供的数据类工具。
# 它特别适合定义这种“主要用来保存数据”的类，
# 可以自动帮我们生成 __init__() 等方法。
from dataclasses import dataclass, field

# Any 表示“这里允许存放任意类型的数据”。
# 后面的工具参数可能是字符串、数字、布尔值等，
# 所以 arguments 使用 Any 保持扩展性。
from typing import Any


# Action 表示：
# “LLM 决定下一步要做什么”。
@dataclass
class Action:

    # tool 保存 LLM 想调用的工具名称。
    # 例如：
    # "read_file"
    # "write_file"
    # "run_terminal"
    tool: str

    # arguments 保存调用这个工具需要的参数。
    #
    # 例如：
    # {
    #     "path": "calculator.py"
    # }
    #
    # field(default_factory=dict)
    # 表示每个 Action 都创建一个新的空字典。
    #
    # 不直接写 arguments = {}，
    # 是为了避免多个对象共享同一个可变字典。
    arguments: dict[str, Any] = field(default_factory=dict)

    # reasoning 保存模型为什么选择这个 Action。
    #
    # 第一版可以使用它观察 Agent 的决策，
    # 后面如果不需要，也可以不让模型返回这个字段。
    reasoning: str = ""


# Observation 表示：
# “执行完 Action 后，环境返回给 Agent 的结果”。
@dataclass
class Observation:

    # content 是工具返回的主要内容。
    #
    # 例如运行 pytest 后：
    # "1 failed, 3 passed"
    #
    # 或读取文件后：
    # "def add(a, b): ..."
    content: str

    # success 表示工具本身是否成功执行。
    #
    # 注意：
    # pytest 测试失败，不一定意味着工具执行失败。
    #
    # 例如：
    # pytest 正常运行并发现测试没通过，
    # run_terminal 工具本身仍然可能是成功执行的。
    success: bool = True

    # metadata 用来保存额外信息。 
    #
    # 例如 Terminal 工具以后可以保存：
    # {
    #     "return_code": 1
    # }
    #
    # 现在暂时不用，但提前留下扩展接口。
    metadata: dict[str, Any] = field(default_factory=dict)


# AgentStep 表示 Agent 完整的一步：
#
# Action
#   ↓
# Tool
#   ↓
# Observation
#
# 把 Action 和 Observation 放在一起，
# 后面就可以完整记录 Agent 的 trajectory。
@dataclass
class AgentStep:

    # 保存这一轮 Agent 做出的决定。
    action: Action

    # 保存执行这个决定之后得到的结果。
    observation: Observation


# AgentState 表示：
# Agent 当前整个任务的运行状态。
#
# 后面的 Agent Loop 会一直读取和修改这个对象。
@dataclass
class AgentState:

    # task 是用户最开始交给 Agent 的任务。
    #
    # 例如：
    # "修复 workspace 中的项目，使所有测试通过。"
    task: str

    # plan 保存任务规划。
    #
    # Planner 目前还没有实现，
    # 所以现在默认是空列表。
    #
    # 后面可能变成：
    # [
    #     "查看项目结构",
    #     "运行测试",
    #     "定位错误",
    #     "修改代码"
    # ]
    plan: list[str] = field(default_factory=list)

    # history 保存 Agent 已经执行过的所有步骤。
    #
    # 这就是最初版本的 trajectory。
    history: list[AgentStep] = field(default_factory=list)

    # step_count 记录已经执行了多少步。
    step_count: int = 0

    # finished 表示任务是否已经结束。
    finished: bool = False

    # final_answer 保存 Agent 最终返回给用户的结果。
    final_answer: str = ""

    # add_step() 用来记录 Agent 新完成的一步。
    def add_step(
        self,
        action: Action,
        observation: Observation,
    ) -> None:

        # 把 Action 和 Observation 组合成一个 AgentStep。
        step = AgentStep(
            action=action,
            observation=observation,
        )

        # 把这一轮记录加入 history。
        self.history.append(step)

        # Agent 执行步数加 1。
        self.step_count += 1

    # finish() 用来告诉 Agent：
    # 当前任务已经完成。
    def finish(self, final_answer: str) -> None:

        # 修改任务状态。
        self.finished = True

        # 保存最终答案。
        self.final_answer = final_answer