# subprocess 是 Python 标准库。
#
# 它用于从 Python 程序中启动外部程序，
# 例如：
#
# python
# pytest
# git
#
# 这是 TerminalTool 最核心的库。
import subprocess


# Path 是 Python 标准库 pathlib 中的路径类。
#
# 我们使用它管理 workspace 路径。
from pathlib import Path


# Any 表示“任意类型”。
#
# Tool 的 arguments 字典中可能保存：
#
# 字符串
# 列表
# 数字
#
# 所以这里使用 Any。
from typing import Any


# 导入所有 Tool 的共同父类。
from swe_agent.tools.base import BaseTool


# 导入统一的工具返回结果 Observation。
from swe_agent.agent.state import Observation


# TerminalTool 负责执行终端命令。
class TerminalTool(BaseTool):

    # 初始化 TerminalTool。
    #
    # workspace 表示：
    #
    # 命令默认在哪个目录下运行。
    def __init__(
        self,
        workspace: Path,
        timeout: int = 30,
    ):

        # resolve() 把路径转换成规范的绝对路径。
        self.workspace = workspace.resolve()

        # 保存最大运行时间。
        #
        # 默认：
        #
        # 30 秒。
        #
        # 防止 Agent 执行某个命令后一直卡住。
        self.timeout = timeout

    # 定义工具名称。
    @property
    def name(self) -> str:

        # 后面 LLM 会通过这个名字调用终端工具。
        return "terminal"

    # 定义工具说明。
    @property
    def description(self) -> str:

        # 后面的 Prompt Builder
        # 会把这段说明提供给 LLM。
        return (
            "在 workspace 中执行终端命令。"
            "参数 command 必须是字符串列表，"
            "例如 ['python', '-m', 'pytest']。"
        )

    # run() 是 Tool 的实际执行入口。
    def run(
        self,
        arguments: dict[str, Any],
    ) -> Observation:

        # 从参数字典中取得 command。
        #
        # 如果不存在，
        # command 就会得到 None。
        command = arguments.get("command")

        # 首先检查 command 必须是 list。
        #
        # 我们暂时不接受：
        #
        # "python -m pytest"
        #
        # 而要求：
        #
        # ["python", "-m", "pytest"]
        if not isinstance(command, list):

            # 参数类型错误时返回失败 Observation。
            return Observation(
                content="terminal 缺少有效的 command 参数",
                success=False,
            )

        # command 不能为空列表。
        if not command:

            # 空命令没有办法执行。
            return Observation(
                content="terminal 的 command 不能为空",
                success=False,
            )

        # 检查 command 中的每一项是否都是字符串。
        #
        # 正确示例：
        #
        # ["python", "-m", "pytest"]
        #
        # 错误示例：
        #
        # ["python", 123]
        if not all(
            isinstance(item, str)
            for item in command
        ):

            # 如果出现非字符串参数，
            # 返回失败。
            return Observation(
                content="terminal 的 command 必须全部由字符串组成",
                success=False,
            )

        # 使用 try / except
        # 捕获执行命令过程中可能发生的异常。
        try:

            # subprocess.run()
            # 会真正启动一个外部程序。
            result = subprocess.run(

                # 要执行的程序和参数。
                #
                # 例如：
                #
                # ["python", "-m", "pytest"]
                command,

                # cwd = current working directory。
                #
                # 指定命令从 workspace 中运行。
                cwd=self.workspace,

                # capture_output=True
                #
                # 表示把：
                #
                # stdout
                # stderr
                #
                # 捕获回来，
                # 而不是直接全部打印到当前终端。
                capture_output=True,

                # text=True
                #
                # 表示把输出直接转换成字符串。
                #
                # 否则得到的会是 bytes。
                text=True,

                # 最多允许命令运行多少秒。
                #
                # 超过这个时间，
                # subprocess 会抛出 TimeoutExpired。
                timeout=self.timeout,
            )

            # stdout 是程序的标准输出。
            stdout = result.stdout

            # stderr 是程序的错误输出。
            stderr = result.stderr

            # returncode 是程序退出码。
            #
            # 通常：
            #
            # 0     = 正常
            # 非 0  = 程序报告错误
            returncode = result.returncode

            # 为了让 LLM 更容易理解，
            # 把 stdout 和 stderr 整理到一起。
            output_parts: list[str] = []

            # 如果 stdout 不为空，
            # 就加入结果。
            if stdout:

                # strip() 去掉字符串首尾多余换行和空格。
                output_parts.append(
                    f"STDOUT:\n{stdout.strip()}"
                )

            # 如果 stderr 不为空，
            # 同样加入结果。
            if stderr:

                output_parts.append(
                    f"STDERR:\n{stderr.strip()}"
                )

            # 如果程序完全没有任何输出，
            # 给 Agent 一个明确提示。
            if not output_parts:

                output_parts.append(
                    "命令执行完成，但没有输出。"
                )

            # "\n\n".join(...)
            #
            # 把多段输出连接成一个字符串。
            output = "\n\n".join(
                output_parts
            )

            # Tool 本身成功完成了一次命令执行。
            #
            # 注意：
            #
            # returncode 不一定为 0。
            #
            # pytest 测试失败时可能 returncode=1，
            # 但 TerminalTool 本身仍然成功执行了 pytest。
            return Observation(
                content=output,
                success=True,

                # 把终端相关的结构化信息
                # 保存到 metadata。
                metadata={

                    # 保存真正执行的命令。
                    "command": command,

                    # 保存程序退出码。
                    "returncode": returncode,

                    # 单独保留 stdout。
                    "stdout": stdout,

                    # 单独保留 stderr。
                    "stderr": stderr,
                },
            )

        # TimeoutExpired 是 subprocess 提供的异常。
        #
        # 当程序运行超过 timeout 秒时发生。
        except subprocess.TimeoutExpired:

            # 告诉 Agent：
            #
            # 这个命令运行时间太长，
            # 已经被终止。
            return Observation(
                content=(
                    f"命令执行超时，"
                    f"超过 {self.timeout} 秒"
                ),
                success=False,
            )

        # FileNotFoundError 表示：
        #
        # 系统找不到要执行的程序。
        #
        # 例如：
        #
        # ["abc123", "xxx"]
        #
        # 而 abc123 根本不存在。
        except FileNotFoundError as error:

            # 返回明确错误信息。
            return Observation(
                content=f"无法找到命令：{str(error)}",
                success=False,
            )

        # 捕获其他普通异常。
        except Exception as error:

            # 把异常信息返回给 Agent。
            return Observation(
                content=f"终端命令执行失败：{str(error)}",
                success=False,
            )
        

# 定义一个专门用于第一次实验的受限终端工具。
# 它继承我们已经实现的 TerminalTool。
class PytestOnlyTerminalTool(TerminalTool):

    # 覆盖原来的工具说明，让 LLM 知道命令限制。
    @property
    def description(self) -> str:

        # 第一版只允许执行 pytest。
        return (
            "在 workspace 中运行 Python 测试。"
            '只允许 command=["python", "-m", "pytest"]。'
        )

    # 覆盖父类的 run() 方法，先检查命令。
    def run(
        self,
        arguments: dict[str, Any],
    ) -> Observation:

        # 从参数字典中获取 LLM 请求执行的命令。
        command = arguments.get("command")

        # 检查命令是否严格符合我们允许的形式。
        if command != ["python", "-m", "pytest"]:

            # 不允许执行其他命令。
            return Observation(
                content="当前只允许执行 python -m pytest",
                success=False,
            )

        # sys 是 Python 标准库。
        # sys.executable 表示当前 Python 解释器的实际路径。
        # 这样可以确保 pytest 使用我们的 uv 环境。
        import sys

        # 调用父类 TerminalTool 的 run()。
        # super() 表示访问父类的方法。
        return super().run(
            {
                # 使用当前 uv 虚拟环境中的 Python。
                "command": [
                    sys.executable,
                    "-m",
                    "pytest",
                ]
            }
        )