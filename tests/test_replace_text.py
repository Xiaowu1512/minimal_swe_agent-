# Path 来自 Python 标准库 pathlib。
# 用于管理文件和目录路径。
from pathlib import Path

# 导入我们刚刚实现的 ReplaceTextTool。
# 它负责对 workspace 内文件执行精确文本替换。
from swe_agent.tools.file_tools import ReplaceTextTool


# 测试 1：
# 验证 ReplaceTextTool 能否正常替换指定文本。
# 同时确保文件中其他代码保持不变。
def test_replace_text_success(tmp_path: Path):

    # pytest 自动提供 tmp_path 临时目录。
    # 在临时目录中创建 workspace。
    workspace = tmp_path / "workspace"

    # 真正创建 workspace 文件夹。
    workspace.mkdir()

    # 定义需要测试的文件路径。
    test_file = workspace / "calculator.py"

    # 创建测试文件。
    # 文件中故意保留一个错误的加法函数。
    # 同时包含一个正确的乘法函数。
    original_content = (
        "def add(a, b):\n"
        "    return a - b\n"
        "\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )

    # 将原始内容写入测试文件。
    test_file.write_text(
        original_content,
        encoding="utf-8",
    )

    # 创建 ReplaceTextTool。
    # 限制它只能访问临时 workspace。
    tool = ReplaceTextTool(
        workspace=workspace
    )

    # 执行文本替换。
    # 将错误的减法修改为加法。
    observation = tool.run(
        {
            "path": "calculator.py",
            "old_text": "return a - b",
            "new_text": "return a + b",
        }
    )

    # 检查工具是否成功执行。
    assert observation.success is True

    # 读取修改后的完整文件内容。
    modified_content = test_file.read_text(
        encoding="utf-8"
    )

    # 检查错误代码是否已经被修复。
    assert "return a + b" in modified_content

    # 检查旧代码是否已经不存在。
    assert "return a - b" not in modified_content

    # 检查其他函数是否仍然存在。
    assert "return a * b" in modified_content

    # 更严格地检查整个文件内容。
    # 确保除了目标文本之外，其他内容完全不变。
    assert modified_content == original_content.replace(
        "return a - b",
        "return a + b",
        1,
    )

    # 检查 metadata 中记录的替换次数。
    assert observation.metadata["replacement_count"] == 1

    # 检查 metadata 中保存的文件路径。
    assert observation.metadata["path"] == "calculator.py"


# 测试 2：
# 验证 old_text 不存在时不会修改文件。
def test_replace_text_not_found(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建文件夹。
    workspace.mkdir()

    # 定义测试文件路径。
    test_file = workspace / "calculator.py"

    # 创建一个内容正确的文件。
    original_content = "def add(a, b):\n    return a + b\n"

    # 写入原始内容。
    test_file.write_text(
        original_content,
        encoding="utf-8",
    )

    # 创建替换工具。
    tool = ReplaceTextTool(
        workspace=workspace
    )

    # 故意替换一个不存在的文本。
    observation = tool.run(
        {
            "path": "calculator.py",
            "old_text": "return a - b",
            "new_text": "return a + b",
        }
    )

    # 因为找不到 old_text，所以应该失败。
    assert observation.success is False

    # 返回的信息应该包含“没有找到”。
    assert "没有找到" in observation.content

    # 再次读取文件，确认内容没有变化。
    assert test_file.read_text(
        encoding="utf-8"
    ) == original_content


# 测试 3：
# 验证 old_text 出现多次时拒绝修改。
# 这是防止 Agent 误修改多个代码位置的重要测试。
def test_replace_text_multiple_matches(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 定义测试文件。
    test_file = workspace / "calculator.py"

    # 故意让同一段代码出现两次。
    original_content = (
        "def add(a, b):\n"
        "    return a - b\n"
        "\n"
        "def another_add(a, b):\n"
        "    return a - b\n"
    )

    # 写入测试文件。
    test_file.write_text(
        original_content,
        encoding="utf-8",
    )

    # 创建替换工具。
    tool = ReplaceTextTool(
        workspace=workspace
    )

    # 尝试替换重复出现的代码。
    observation = tool.run(
        {
            "path": "calculator.py",
            "old_text": "return a - b",
            "new_text": "return a + b",
        }
    )

    # 由于出现两次，必须拒绝替换。
    assert observation.success is False

    # 检查错误信息是否说明重复次数。
    assert "出现了 2 次" in observation.content

    # 检查文件是否完全没有被修改。
    assert test_file.read_text(
        encoding="utf-8"
    ) == original_content


# 测试 4：
# 验证必要参数缺失或类型错误时能够正确处理。
def test_replace_text_invalid_arguments(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建替换工具。
    tool = ReplaceTextTool(
        workspace=workspace
    )

    # 情况一：
    # 缺少 path 参数。
    observation_no_path = tool.run(
        {
            "old_text": "a",
            "new_text": "b",
        }
    )

    # 应该返回失败。
    assert observation_no_path.success is False

    # 检查 path 参数错误提示。
    assert "path 参数" in observation_no_path.content

    # 情况二：
    # old_text 为空字符串。
    observation_empty_old = tool.run(
        {
            "path": "calculator.py",
            "old_text": "",
            "new_text": "hello",
        }
    )

    # 空旧文本不允许替换。
    assert observation_empty_old.success is False

    # 检查 old_text 参数错误提示。
    assert "old_text 参数" in observation_empty_old.content

    # 情况三：
    # new_text 使用整数，而不是字符串。
    observation_invalid_new = tool.run(
        {
            "path": "calculator.py",
            "old_text": "hello",
            "new_text": 123,
        }
    )

    # 参数类型错误，应该返回失败。
    assert observation_invalid_new.success is False

    # 检查 new_text 参数错误提示。
    assert "new_text 参数" in observation_invalid_new.content


# 测试 5：
# 验证目标文件不存在时能够正确返回失败。
def test_replace_text_file_not_exists(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建替换工具。
    tool = ReplaceTextTool(
        workspace=workspace
    )

    # 指定一个不存在的文件。
    observation = tool.run(
        {
            "path": "not_exist.py",
            "old_text": "hello",
            "new_text": "world",
        }
    )

    # 文件不存在，所以替换应该失败。
    assert observation.success is False

    # 检查是否返回“文件不存在”。
    assert "文件不存在" in observation.content

    # 确认工具没有意外创建文件。
    assert (workspace / "not_exist.py").exists() is False


# 测试 6：
# 验证 ReplaceTextTool 不能修改 workspace 之外的文件。
def test_replace_text_cannot_escape_workspace(tmp_path: Path):

    # 创建临时 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 在 workspace 外部创建一个测试文件。
    # 目录结构：
    #
    # tmp_path/
    # ├── secret.txt
    # └── workspace/
    outside_file = tmp_path / "secret.txt"

    # 定义外部文件的原始内容。
    original_content = "secret information"

    # 向外部文件写入内容。
    outside_file.write_text(
        original_content,
        encoding="utf-8",
    )

    # 创建替换工具。
    tool = ReplaceTextTool(
        workspace=workspace
    )

    # ../ 表示返回上一级目录。
    # 这里故意尝试修改 workspace 外部的文件。
    observation = tool.run(
        {
            "path": "../secret.txt",
            "old_text": "secret",
            "new_text": "modified",
        }
    )

    # 越界访问必须失败。
    assert observation.success is False

    # 检查是否返回安全限制提示。
    assert "禁止访问 workspace 之外的文件" in observation.content

    # 检查外部文件的内容是否保持不变。
    assert outside_file.read_text(
        encoding="utf-8"
    ) == original_content