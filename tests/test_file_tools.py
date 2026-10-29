# Path 来自 Python 标准库 pathlib。
# 它用于以面向对象的方式处理文件和目录路径。
from pathlib import Path


# pytest 是我们的测试框架。
# 这里虽然没有直接写 pytest.xxx，
# 但会使用 pytest 自动提供的 tmp_path 测试夹具。
import pytest


# 导入我们刚刚实现的 ReadFileTool。
# 导入我们已经实现的三个文件工具。
from swe_agent.tools.file_tools import (
    ReadFileTool,
    WriteFileTool,
    SearchTextTool,
)

# 测试 1：
# ReadFileTool 能不能正常读取 workspace 中的文件。
#
# tmp_path 是 pytest 自动提供的 fixture（测试夹具）。
# pytest 会为这个测试临时创建一个文件夹，
# 测试结束后我们不用自己管理它。
def test_read_file_success(tmp_path: Path):

    # 在 pytest 提供的临时目录中创建 workspace 文件夹。
    #
    # 例如可能实际创建：
    #
    # C:\Users\...\Temp\pytest-xxx\workspace
    workspace = tmp_path / "workspace"

    # mkdir() 是 Path 的方法，
    # 用来真正创建目录。
    workspace.mkdir()

    # 在 workspace 中定义一个测试文件路径。
    test_file = workspace / "calculator.py"

    # write_text() 是 Path 的方法，
    # 用来向文本文件写入内容。
    #
    # encoding="utf-8"
    # 表示使用 UTF-8 编码。
    test_file.write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    # 创建 ReadFileTool。
    #
    # 告诉这个工具：
    # 它只能操作刚才创建的 workspace。
    tool = ReadFileTool(
        workspace=workspace
    )

    # 调用工具。
    #
    # 注意这里传入的是相对于 workspace 的路径，
    # 而不是完整路径。
    observation = tool.run(
        {
            "path": "calculator.py"
        }
    )

    # 检查工具是否执行成功。
    assert observation.success is True

    # 检查读取到的文件内容是否正确。
    assert observation.content == (
        "def add(a, b):\n"
        "    return a + b\n"
    )

    # 检查 metadata 中是否正确记录了文件路径。
    assert observation.metadata["path"] == "calculator.py"


# 测试 2：
# 尝试读取一个不存在的文件。
def test_read_file_not_exists(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 真正创建 workspace 目录。
    workspace.mkdir()

    # 创建 ReadFileTool。
    tool = ReadFileTool(
        workspace=workspace
    )

    # 请求读取一个不存在的文件。
    observation = tool.run(
        {
            "path": "not_exist.py"
        }
    )

    # 文件不存在，
    # 所以工具应该返回失败。
    assert observation.success is False

    # 检查返回信息中是否包含“文件不存在”。
    #
    # in 表示：
    # 判断左边字符串是否包含在右边字符串中。
    assert "文件不存在" in observation.content


# 测试 3：
# 检查 ReadFileTool 是否能够阻止访问 workspace 外部。
def test_read_file_cannot_escape_workspace(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 在 workspace 外面创建一个文件。
    #
    # 注意：
    #
    # tmp_path/
    # ├── secret.txt
    # └── workspace/
    #
    # secret.txt 并不属于 workspace。
    outside_file = tmp_path / "secret.txt"

    # 给这个外部文件写入内容。
    outside_file.write_text(
        "secret information",
        encoding="utf-8",
    )

    # 创建 ReadFileTool。
    tool = ReadFileTool(
        workspace=workspace
    )

    # "../secret.txt"
    # 表示：
    #
    # 从 workspace 返回上一级目录，
    # 然后尝试读取 secret.txt。
    #
    # 这正是我们希望阻止的路径越界行为。
    observation = tool.run(
        {
            "path": "../secret.txt"
        }
    )

    # 因为访问了 workspace 外部，
    # 所以应该失败。
    assert observation.success is False

    # 检查错误信息。
    assert "禁止访问 workspace 之外的文件" in observation.content


# 测试 4：
# 检查没有提供 path 参数时是否能够正确处理。
def test_read_file_without_path(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 ReadFileTool。
    tool = ReadFileTool(
        workspace=workspace
    )

    # 故意传入空字典，
    # 也就是没有：
    #
    # {"path": "..."}
    observation = tool.run({})

    # 参数缺失时应该执行失败。
    assert observation.success is False

    # 检查错误提示。
    assert "缺少有效的 path 参数" in observation.content


'WriteFileTool测试：'
# 测试 1：
# WriteFileTool 能不能创建一个原本不存在的新文件。
def test_write_file_create_new_file(tmp_path: Path):

    # 在 pytest 提供的临时目录中定义 workspace 路径。
    workspace = tmp_path / "workspace"

    # 创建 workspace 文件夹。
    workspace.mkdir()

    # 创建 WriteFileTool，
    # 并限制它只能操作这个 workspace。
    tool = WriteFileTool(
        workspace=workspace
    )

    # 调用 write_file 工具。
    observation = tool.run(
        {
            # 指定要创建的文件。
            "path": "calculator.py",

            # 指定文件完整内容。
            "content": "def add(a, b):\n    return a + b\n",
        }
    )

    # 检查工具是否执行成功。
    assert observation.success is True

    # 得到刚刚应该创建出来的文件路径。
    created_file = workspace / "calculator.py"

    # exists() 判断文件是否真的存在。
    assert created_file.exists()

    # read_text() 再把文件读出来，
    # 检查实际写入的内容是否正确。
    assert created_file.read_text(
        encoding="utf-8"
    ) == "def add(a, b):\n    return a + b\n"

    # 因为这个文件之前不存在，
    # 所以 existed_before 应该是 False。
    assert observation.metadata["existed_before"] is False


# 测试 2：
# WriteFileTool 能不能覆盖一个已经存在的文件。
def test_write_file_overwrite_existing_file(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建 workspace 文件夹。
    workspace.mkdir()

    # 定义一个已经存在的文件。
    target_file = workspace / "calculator.py"

    # 先给文件写入一段错误代码。
    target_file.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    # 创建 WriteFileTool。
    tool = WriteFileTool(
        workspace=workspace
    )

    # 使用 WriteFileTool 写入新的完整内容。
    observation = tool.run(
        {
            "path": "calculator.py",
            "content": "def add(a, b):\n    return a + b\n",
        }
    )

    # 工具应该执行成功。
    assert observation.success is True

    # 再读取文件，
    # 确认旧内容已经被新内容覆盖。
    assert target_file.read_text(
        encoding="utf-8"
    ) == "def add(a, b):\n    return a + b\n"

    # 因为文件原本已经存在，
    # existed_before 应该是 True。
    assert observation.metadata["existed_before"] is True


# 测试 3：
# 检查没有提供必要参数时是否能够正确返回失败。
def test_write_file_missing_arguments(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 WriteFileTool。
    tool = WriteFileTool(
        workspace=workspace
    )

    # 情况一：
    # 没有提供 path，
    # 只提供 content。
    observation_without_path = tool.run(
        {
            "content": "hello"
        }
    )

    # 应该执行失败。
    assert observation_without_path.success is False

    # 检查错误信息中是否提示 path 参数有问题。
    assert "path 参数" in observation_without_path.content

    # 情况二：
    # 提供 path，
    # 但没有提供 content。
    observation_without_content = tool.run(
        {
            "path": "test.txt"
        }
    )

    # 同样应该执行失败。
    assert observation_without_content.success is False

    # 检查错误信息中是否提示 content 参数有问题。
    assert "content 参数" in observation_without_content.content


# 测试 4：
# 检查 WriteFileTool 是否能够阻止写入 workspace 外部。
def test_write_file_cannot_escape_workspace(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 WriteFileTool。
    tool = WriteFileTool(
        workspace=workspace
    )

    # 尝试使用 "../"
    # 从 workspace 返回上一级目录，
    # 再创建 secret.txt。
    #
    # 这是我们明确不允许的操作。
    observation = tool.run(
        {
            "path": "../secret.txt",
            "content": "secret information",
        }
    )

    # 应该执行失败。
    assert observation.success is False

    # 检查是否返回了越界访问提示。
    assert "禁止访问 workspace 之外的文件" in observation.content

    # 进一步检查 workspace 外部
    # 是否真的没有被创建出 secret.txt。
    outside_file = tmp_path / "secret.txt"

    # 文件应该不存在。
    assert outside_file.exists() is False


'SearchTextTool：'
# 测试 1：
# 检查 SearchTextTool 能否在单个文件中正常找到指定文本。
def test_search_text_success(tmp_path: Path):

    # pytest 自动给我们提供一个临时目录。
    # 我们在这个临时目录中创建 workspace。
    workspace = tmp_path / "workspace"

    # 真正创建 workspace 文件夹。
    workspace.mkdir()

    # 在 workspace 中创建一个 Python 文件。
    test_file = workspace / "calculator.py"

    # 给测试文件写入几行代码。
    test_file.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n",
        encoding="utf-8",
    )

    # 创建 SearchTextTool。
    #
    # 告诉它只能在这个临时 workspace 中搜索。
    tool = SearchTextTool(
        workspace=workspace
    )

    # 搜索文本 "def add"。
    #
    # path="."
    # 表示从整个 workspace 开始搜索。
    observation = tool.run(
        {
            "query": "def add",
            "path": ".",
        }
    )

    # 搜索过程应该正常完成。
    assert observation.success is True

    # 返回结果中应该包含文件名。
    assert "calculator.py" in observation.content

    # 返回结果中应该包含我们搜索到的代码。
    assert "def add(a, b):" in observation.content

    # 搜索结果数量应该是 1。
    assert observation.metadata["match_count"] == 1

    # metadata 中应该保存原始 query。
    assert observation.metadata["query"] == "def add"


# 测试 2：
# 检查 SearchTextTool 能否递归搜索多个文件和子目录。
def test_search_text_multiple_files(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建 workspace 目录。
    workspace.mkdir()

    # 创建第一个 Python 文件。
    first_file = workspace / "main.py"

    # 写入包含目标文本的代码。
    first_file.write_text(
        "print('hello')\n",
        encoding="utf-8",
    )

    # 创建一个子目录。
    utils_directory = workspace / "utils"

    # 创建这个子目录。
    utils_directory.mkdir()

    # 在子目录中创建第二个文件。
    second_file = utils_directory / "helper.py"

    # 第二个文件中也包含 hello。
    second_file.write_text(
        "message = 'hello'\n",
        encoding="utf-8",
    )

    # 再创建一个不包含 hello 的文件。
    third_file = workspace / "other.py"

    # 写入其他内容。
    third_file.write_text(
        "value = 123\n",
        encoding="utf-8",
    )

    # 创建搜索工具。
    tool = SearchTextTool(
        workspace=workspace
    )

    # 从整个 workspace 搜索 "hello"。
    observation = tool.run(
        {
            "query": "hello",
            "path": ".",
        }
    )

    # 搜索应该正常执行。
    assert observation.success is True

    # main.py 中存在一个 hello。
    assert "main.py" in observation.content

    # helper.py 中也存在一个 hello。
    assert "helper.py" in observation.content

    # 所以应该找到两个匹配结果。
    assert observation.metadata["match_count"] == 2


# 测试 3：
# 搜索正常完成，但是没有找到任何内容。
def test_search_text_no_match(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建一个测试文件。
    test_file = workspace / "calculator.py"

    # 文件中没有 "multiply"。
    test_file.write_text(
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )

    # 创建 SearchTextTool。
    tool = SearchTextTool(
        workspace=workspace
    )

    # 搜索一个不存在的字符串。
    observation = tool.run(
        {
            "query": "multiply",
            "path": ".",
        }
    )

    # 注意这里非常重要：
    #
    # 没搜到内容 ≠ 工具执行失败。
    #
    # SearchTextTool 本身成功完成了搜索，
    # 所以 success 仍然应该是 True。
    assert observation.success is True

    # 返回结果应该告诉我们没有找到。
    assert "未找到文本" in observation.content

    # 匹配数量应该是 0。
    assert observation.metadata["match_count"] == 0


# 测试 4：
# 搜索一个根本不存在的目录时，
# 应该返回失败结果。
def test_search_text_path_not_exists(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 SearchTextTool。
    tool = SearchTextTool(
        workspace=workspace
    )

    # 指定一个不存在的搜索目录。
    observation = tool.run(
        {
            "query": "hello",
            "path": "not_exist",
        }
    )

    # 因为搜索路径本身不存在，
    # 所以这次工具执行应该失败。
    assert observation.success is False

    # 检查错误信息。
    assert "搜索路径不存在" in observation.content


# 测试 5：
# 检查 SearchTextTool 是否能够阻止 Agent
# 搜索 workspace 外部的文件。
def test_search_text_cannot_escape_workspace(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建 workspace。
    workspace.mkdir()

    # 在 workspace 外部创建一个文件。
    #
    # 目录结构类似：
    #
    # tmp_path/
    # ├── secret.txt
    # └── workspace/
    outside_file = tmp_path / "secret.txt"

    # 给外部文件写入一些内容。
    outside_file.write_text(
        "very secret information",
        encoding="utf-8",
    )

    # 创建 SearchTextTool。
    tool = SearchTextTool(
        workspace=workspace
    )

    # "../secret.txt"
    # 表示尝试跳出 workspace，
    # 去访问上一级目录中的文件。
    observation = tool.run( 
        {
            "query": "secret",
            "path": "../secret.txt",
        }
    )

    # 这种操作必须被拒绝。
    assert observation.success is False

    # 检查是否返回了安全限制提示。
    assert "禁止访问 workspace 之外的文件" in observation.content