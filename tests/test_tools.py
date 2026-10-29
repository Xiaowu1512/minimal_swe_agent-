# pytest 是我们已经安装的测试框架。
# 这里主要使用 pytest.raises() 来测试“是否正确抛出异常”。
import pytest


# 导入 BaseTool。
# 我们会先创建一个假的 Tool，用它来测试 ToolRegistry。
from swe_agent.tools.base import BaseTool


# 导入 ToolRegistry。
# 这是我们这一阶段真正要测试的对象。
from swe_agent.tools.registry import ToolRegistry


# 导入 Observation。
# 因为所有 Tool 的 run() 都必须返回 Observation。
from swe_agent.agent.state import Observation


# 创建一个专门用于测试的假工具。
#
# 它继承 BaseTool，
# 所以必须实现：
#
# name
# description
# run()
class FakeTool(BaseTool):

    # 工具名称。
    @property
    def name(self) -> str:

        # 返回固定名称 fake_tool。
        return "fake_tool"

    # 工具说明。
    @property
    def description(self) -> str:

        # 这个说明暂时只用于测试。
        return "这是一个测试工具"

    # 实现 BaseTool 规定的 run() 方法。
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
# 验证 ToolRegistry 能否正常注册并获取一个 Tool。
def test_register_and_get_tool():

    # 创建一个空的工具注册中心。
    registry = ToolRegistry()

    # 创建我们的假工具。
    tool = FakeTool()

    # 把假工具注册进去。
    registry.register(tool)

    # 根据工具名称把它取出来。
    retrieved_tool = registry.get("fake_tool")

    # 检查取出来的对象，
    # 是否就是刚才注册进去的那个对象。
    assert retrieved_tool is tool

    # 检查 names() 是否正确返回工具名称。
    assert registry.names() == ["fake_tool"]

    # 检查 all() 是否正确返回工具对象。
    assert registry.all() == [tool]


# 测试 2：
# 验证重复注册同名工具时是否会报错。
def test_duplicate_tool_raises_error():

    # 创建 ToolRegistry。
    registry = ToolRegistry()

    # 创建第一个假工具。
    tool1 = FakeTool()

    # 创建第二个假工具。
    # 虽然是两个不同对象，
    # 但它们的 name 都是 "fake_tool"。
    tool2 = FakeTool()

    # 第一次注册应该正常成功。
    registry.register(tool1)

    # pytest.raises(ValueError)
    # 表示：
    #
    # “下面这段代码应该抛出 ValueError。”
    #
    # 如果没有抛出，
    # 反而说明测试失败。
    with pytest.raises(ValueError):

        # 重复注册同名工具。
        registry.register(tool2)


# 测试 3：
# 验证获取一个不存在的工具时是否会报错。
def test_get_unknown_tool_raises_error():

    # 创建空的 ToolRegistry。
    registry = ToolRegistry()

    # 我们预期 registry.get()
    # 会因为工具不存在而抛出 KeyError。
    with pytest.raises(KeyError):

        # 尝试获取一个从未注册过的工具。
        registry.get("not_exist")