# pytest 是我们已经安装的测试框架。
# 这里主要使用 pytest.raises() 来测试异常。
import pytest

# 导入我们刚刚实现的 ActionParser。
from swe_agent.parser.action_parser import ActionParser


# 测试 1：
# 检查格式正确的 JSON 是否可以正常解析成 Action。
def test_parse_valid_action():

    # 创建 ActionParser 对象。
    parser = ActionParser()

    # 模拟 LLM 返回的 JSON 字符串。
    text = """
    {
        "reasoning": "需要先读取 calculator.py",
        "tool": "read_file",
        "arguments": {
            "path": "calculator.py"
        }
    }
    """

    # 调用 parse()，把字符串转换成 Action。
    action = parser.parse(text)

    # 检查 tool 是否解析正确。
    assert action.tool == "read_file"

    # 检查 arguments 是否解析正确。
    assert action.arguments["path"] == "calculator.py"

    # 检查 reasoning 是否解析正确。
    assert action.reasoning == "需要先读取 calculator.py"


# 测试 2：
# 检查没有 arguments 和 reasoning 时，
# Parser 是否会自动使用默认值。
def test_parse_without_arguments():

    # 创建 Parser。
    parser = ActionParser()

    # 这里只提供 tool。
    text = """
    {
        "tool": "terminal"
    }
    """

    # 执行解析。
    action = parser.parse(text)

    # 检查 tool。
    assert action.tool == "terminal"

    # 没有 arguments 时应该得到空字典。
    assert action.arguments == {}

    # 没有 reasoning 时应该得到空字符串。
    assert action.reasoning == ""


# 测试 3：
# 检查 LLM 返回 Markdown JSON 代码块时，
# Parser 是否仍然能够正常解析。
def test_parse_markdown_json_block():

    # 创建 Parser。
    parser = ActionParser()

    # 模拟大模型常见的 ```json ... ``` 输出格式。
    text = """
```json
{
    "tool": "read_file",
    "arguments": {
        "path": "main.py"
    }
}
```
"""

    # 执行解析。
    action = parser.parse(text)

    # 检查工具名称。
    assert action.tool == "read_file"

    # 检查文件路径。
    assert action.arguments["path"] == "main.py"


# 测试 4：
# 检查非法 JSON 是否会正确抛出 ValueError。
def test_parse_invalid_json():

    # 创建 Parser。
    parser = ActionParser()

    # 故意构造错误 JSON。
    text = """
    {
        tool: read_file
    }
    """

    # pytest.raises(ValueError) 表示：
    # 预期下面代码会抛出 ValueError。
    with pytest.raises(ValueError):

        # 尝试解析非法 JSON。
        parser.parse(text)


# 测试 5：
# 检查缺少 tool 字段时，
# Parser 是否会正确抛出 ValueError。
def test_parse_missing_tool():

    # 创建 Parser。
    parser = ActionParser()

    # 故意不提供 tool。
    text = """
    {
        "arguments": {
            "path": "main.py"
        }
    }
    """

    # 缺少 tool 时应该抛出 ValueError。
    with pytest.raises(ValueError):

        # 执行解析。
        parser.parse(text)