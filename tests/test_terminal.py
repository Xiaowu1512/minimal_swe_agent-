# sys 是 Python 标准库。
# 这里使用 sys.executable 获取“当前正在运行测试的 Python 解释器路径”。
#
# 这样比直接写 "python" 更稳，
# 因为可以确保 subprocess 使用的是当前 uv 虚拟环境里的 Python。
import sys


# Path 用来表示文件和目录路径。
from pathlib import Path


# 导入我们刚刚实现的 TerminalTool。
from swe_agent.tools.terminal import TerminalTool


# 测试 1：
# 检查 TerminalTool 能否正常执行一个简单的 Python 命令。
def test_terminal_success(tmp_path: Path):

    # 创建一个临时 workspace。
    workspace = tmp_path / "workspace"

    # 真正创建目录。
    workspace.mkdir()

    # 创建 TerminalTool。
    tool = TerminalTool(
        workspace=workspace
    )

    # 执行：
    #
    # python -c "print('hello')"
    #
    # -c 表示：
    # 让 Python 直接执行后面的字符串代码。
    observation = tool.run(
        {
            "command": [
                sys.executable,
                "-c",
                "print('hello')",
            ]
        }
    )

    # TerminalTool 应该正常执行完成。
    assert observation.success is True

    # 标准输出中应该包含 hello。
    assert "hello" in observation.content

    # 程序正常退出，
    # returncode 应该是 0。
    assert observation.metadata["returncode"] == 0


# 测试 2：
# 检查“程序返回非 0”时，
# TerminalTool 本身仍然算执行成功。
def test_terminal_nonzero_returncode(tmp_path: Path):

    # 创建 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 TerminalTool。
    tool = TerminalTool(
        workspace=workspace
    )

    # sys.exit(3)
    # 会让 Python 程序主动以退出码 3 结束。
    #
    # 这模拟：
    #
    # pytest 成功运行，
    # 但是测试失败，
    # 所以返回非 0 returncode。
    observation = tool.run(
        {
            "command": [
                sys.executable,
                "-c",
                "import sys; sys.exit(3)",
            ]
        }
    )

    # TerminalTool 确实成功启动并执行了程序，
    # 所以 success 仍然应该是 True。
    assert observation.success is True

    # 但是程序自己的退出码应该是 3。
    assert observation.metadata["returncode"] == 3


# 测试 3：
# 检查没有提供正确 command 参数时的处理。
def test_terminal_invalid_command(tmp_path: Path):

    # 创建 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 TerminalTool。
    tool = TerminalTool(
        workspace=workspace
    )

    # 故意传入字符串，
    # 而不是我们规定的字符串列表。
    observation = tool.run(
        {
            "command": "python test.py"
        }
    )

    # 参数格式错误，
    # Tool 应该返回失败。
    assert observation.success is False

    # 检查错误提示。
    assert "command 参数" in observation.content


# 测试 4：
# 检查执行一个根本不存在的程序时是否能够正确处理。
def test_terminal_command_not_found(tmp_path: Path):

    # 创建 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 TerminalTool。
    tool = TerminalTool(
        workspace=workspace
    )

    # 使用一个几乎不可能存在的程序名。
    observation = tool.run(
        {
            "command": [
                "this_command_should_not_exist_123456"
            ]
        }
    )

    # 系统找不到程序，
    # Tool 应该执行失败。
    assert observation.success is False

    # 检查错误信息。
    assert "无法找到命令" in observation.content


# 测试 5：
# 检查 timeout 是否真正有效。
def test_terminal_timeout(tmp_path: Path):

    # 创建 workspace。
    workspace = tmp_path / "workspace"

    # 创建目录。
    workspace.mkdir()

    # 创建 TerminalTool。
    #
    # 这里故意把 timeout 设置得非常短：
    # 0.1 秒。
    tool = TerminalTool(
        workspace=workspace,
        timeout=0.1,
    )

    # 执行一个会睡眠 2 秒的 Python 程序。
    #
    # time.sleep(2)
    # 表示程序暂停 2 秒。
    observation = tool.run(
        {
            "command": [
                sys.executable,
                "-c",
                "import time; time.sleep(2)",
            ]
        }
    )

    # 因为允许执行的时间只有 0.1 秒，
    # 所以应该超时失败。
    assert observation.success is False

    # 检查超时信息。
    assert "超时" in observation.content