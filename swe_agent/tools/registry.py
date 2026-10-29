# 导入 BaseTool。
# ToolRegistry 只负责管理继承自 BaseTool 的工具。
from swe_agent.tools.base import BaseTool


# 定义工具注册中心。
#
# 以后所有工具，例如：
#
# read_file
# write_file
# run_terminal
#
# 都会先注册到这里。
class ToolRegistry:

    # __init__() 是初始化方法。
    # 创建 ToolRegistry() 对象时自动执行。
    def __init__(self):

        # _tools 用一个字典保存所有已经注册的工具。
        #
        # 字典结构大致会是：
        #
        # {
        #     "read_file": ReadFileTool对象,
        #     "write_file": WriteFileTool对象,
        # }
        #
        # dict[str, BaseTool] 表示：
        # key 是字符串，
        # value 是 BaseTool 类型的对象。
        self._tools: dict[str, BaseTool] = {}

    # register() 用来注册一个工具。
    #
    # 例如：
    #
    # registry.register(ReadFileTool())
    def register(self, tool: BaseTool) -> None:

        # tool.name 就是工具自己的名字。
        #
        # 例如：
        #
        # "read_file"
        tool_name = tool.name

        # 如果已经存在同名工具，
        # 这里直接报错，避免后面的工具被悄悄覆盖。
        if tool_name in self._tools:

            # ValueError 是 Python 内置异常类型，
            # 表示“传入的值不符合要求”。
            raise ValueError(
                f"工具 '{tool_name}' 已经注册"
            )

        # 把工具保存进字典。
        #
        # 例如最终可能变成：
        #
        # self._tools["read_file"] = ReadFileTool()
        self._tools[tool_name] = tool

    # get() 根据工具名称获取对应的工具对象。
    #
    # 例如：
    #
    # tool = registry.get("read_file")
    def get(self, name: str) -> BaseTool:

        # 如果这个工具没有注册，
        # 就立即报错。
        if name not in self._tools:

            # KeyError 是 Python 内置异常，
            # 通常表示：
            # “在字典或映射结构中找不到这个 key。”
            raise KeyError(
                f"没有找到工具 '{name}'"
            )

        # 返回已经注册的工具对象。
        return self._tools[name]

    # names() 返回目前所有工具的名字。
    #
    # 后面的 Prompt Builder 可以使用它，
    # 告诉 LLM 当前有哪些工具可以调用。
    def names(self) -> list[str]:

        # self._tools.keys()
        # 会得到字典中所有 key。
        #
        # list() 把它转换为普通 Python 列表。
        return list(self._tools.keys())

    # all() 返回所有已经注册的工具对象。
    #
    # 后面生成工具描述时会用到。
    def all(self) -> list[BaseTool]:

        # self._tools.values()
        # 获取字典中所有 value，
        # 也就是所有 Tool 对象。
        return list(self._tools.values())