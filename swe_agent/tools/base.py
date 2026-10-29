# ABC 和 abstractmethod 来自 Python 标准库 abc。
#
# ABC = Abstract Base Class，抽象基类。
# abstractmethod 用来规定：
# “所有继承 BaseTool 的具体工具，都必须实现某些方法或属性。”
from abc import ABC, abstractmethod


# Any 来自 Python 标准库 typing。
#
# Any 表示“任意类型”。
# 工具参数以后可能是字符串、数字、布尔值等，
# 所以这里允许参数字典中保存不同类型的数据。
from typing import Any


# 导入我们阶段 2 已经定义好的 Observation。
#
# Tool 执行完成之后，不直接随便返回字符串，
# 而是统一返回 Observation。
from swe_agent.agent.state import Observation


# BaseTool 是所有工具的共同父类。
#
# 例如以后：
#
# ReadFileTool
# WriteFileTool
# SearchTextTool
# TerminalTool
#
# 都要继承 BaseTool。
class BaseTool(ABC):

    # @property 是 Python 自带的装饰器。
    #
    # 它可以让一个方法像普通属性一样访问。
    #
    # 例如本来可能要写：
    #
    # tool.name()
    #
    # 使用 @property 后可以直接写：
    #
    # tool.name
    @property

    # @abstractmethod 表示：
    # 所有 BaseTool 的子类都必须实现 name。
    @abstractmethod
    def name(self) -> str:

        # name 用来表示工具的唯一名称。
        #
        # 例如：
        #
        # "read_file"
        # "write_file"
        # "run_terminal"
        #
        # 后面的 ToolRegistry 会通过这个名字寻找工具。
        pass

    # description 同样设计成属性。
    @property

    # 所有具体工具都必须提供自己的说明。
    @abstractmethod
    def description(self) -> str:

        # description 是给 LLM 看的工具说明。
        #
        # 例如 read_file 可能描述为：
        #
        # "读取 workspace 中指定文件的内容"
        #
        # 后面 Prompt Builder 会把这些工具说明
        # 告诉大语言模型。
        pass

    # run() 是所有工具最核心的方法。
    #
    # Agent 找到工具以后，
    # 最终都会通过 tool.run(...) 执行。
    @abstractmethod
    def run(
        self,

        # arguments 是调用工具所需要的参数。
        #
        # 例如 read_file：
        #
        # {
        #     "path": "calculator.py"
        # }
        #
        # dict[str, Any] 表示：
        # key 必须是字符串，
        # value 可以是任意类型。
        arguments: dict[str, Any],

    # -> Observation 表示：
    # 所有工具执行结束后，
    # 必须统一返回 Observation 对象。
    ) -> Observation:

        # BaseTool 本身不知道具体工具如何执行，
        # 所以这里只规定接口，不实现逻辑。 
        pass