# Path 来自 Python 标准库 pathlib。
# 它专门用来处理文件路径，比直接操作字符串更安全、清晰。
from pathlib import Path


# Any 表示参数的值可以是任意类型。
from typing import Any


# 导入所有工具共同的父类 BaseTool。
from swe_agent.tools.base import BaseTool


# 导入工具统一返回的数据结构 Observation。
from swe_agent.agent.state import Observation


# FileToolBase 是所有“文件类工具”的共同父类。
#
# 后面的：
#
# ReadFileTool
# WriteFileTool
# SearchTextTool
#
# 都可以继承它。
#
# 它最重要的职责是：
# 确保所有文件操作只能发生在 workspace 中。
class FileToolBase(BaseTool):

    # 创建文件工具时，需要告诉它 workspace 在哪里。
    def __init__(self, workspace: Path):

        # resolve() 会把路径转换
        # 成规范的绝对路径。
        #
        # 例如：
        #
        # workspace
        #
        # 可能最终变成：
        #
        # C:\Users\...\minimal_swe_agent\workspace
        self.workspace = workspace.resolve()

    # 定义一个内部辅助方法。
    #
    # "_" 开头表示：
    # 这个方法主要供类内部使用。
    def _resolve_path(self, relative_path: str) -> Path:

        # 把 workspace 和模型给出的相对路径拼接起来。
        #
        # 例如：
        #
        # workspace
        # +
        # calculator/main.py
        #
        # 得到：
        #
        # workspace/calculator/main.py
        target_path = (self.workspace / relative_path).resolve()

        # is_relative_to() 是 Path 提供的方法。
        #
        # 用来判断 target_path 是否仍然位于 workspace 内部。
        #
        # 这是一个很重要的安全限制。
        if not target_path.is_relative_to(self.workspace):

            # 如果模型尝试：
            #
            # ../../其他文件
            #
            # 就会走到这里。
            raise ValueError(
                "禁止访问 workspace 之外的文件"
            )

        # 返回已经检查过的安全路径。
        return target_path


# ReadFileTool 是真正负责读取文件的工具。
class ReadFileTool(FileToolBase):

    # 定义这个工具的唯一名称。
    @property
    def name(self) -> str:

        # 后面 LLM 想读取文件时，
        # 会选择名为 read_file 的工具。
        return "read_file"

    # 定义工具的说明。
    @property
    def description(self) -> str:

        # 后面的 Prompt Builder 会把这个说明告诉 LLM。
        return (
            "读取 workspace 中指定文本文件的内容。"
            "参数：path，表示相对于 workspace 的文件路径。"
        )

    # 实现真正的文件读取操作。
    def run(
        self,
        arguments: dict[str, Any],
    ) -> Observation:

        # 使用 get() 从参数字典中读取 path。
        #
        # 如果模型没有提供 path，
        # get() 会返回 None。
        relative_path = arguments.get("path")

        # 首先检查 path 是否存在，
        # 并且必须是字符串。
        if not isinstance(relative_path, str):

            # 参数错误时返回失败的 Observation。
            return Observation(
                content="read_file 缺少有效的 path 参数",
                success=False,
            )
 
        # try/except 用来捕获文件操作过程中可能出现的异常。
        try:

            # 调用父类提供的方法，
            # 得到经过安全检查的真实路径。
            target_path = self._resolve_path(relative_path)

            # exists() 判断路径是否存在。
            if not target_path.exists():

                # 文件不存在时返回失败结果。
                return Observation(
                    content=f"文件不存在：{relative_path}",
                    success=False,
                )

            # is_file() 判断这个路径是不是普通文件。
            #
            # 防止把文件夹当成文件读取。
            if not target_path.is_file():

                # 如果路径不是文件，返回失败。
                return Observation(
                    content=f"目标不是文件：{relative_path}",
                    success=False,
                )

            # read_text() 是 Path 提供的文本读取方法。
            #
            # encoding="utf-8"
            # 指定使用 UTF-8 编码读取。
            content = target_path.read_text(
                encoding="utf-8"
            )

            # 文件读取成功后，
            # 把内容包装成统一的 Observation 返回。
            return Observation(
                content=content,
                success=True,

                # metadata 保存额外信息。
                metadata={
                    "path": relative_path,
                },
            )

        # 捕获上面可能发生的异常。
        except Exception as error:

            # str(error)
            # 把异常对象转换成方便阅读的文字。
            return Observation(
                content=f"读取文件失败：{str(error)}",
                success=False,
            )

# WriteFileTool 用来向 workspace 中的文件写入内容。
#
# 它同样继承 FileToolBase，
# 因此自动拥有：
#
# 1. workspace 路径管理
# 2. _resolve_path() 路径解析
# 3. 禁止访问 workspace 外部
class WriteFileTool(FileToolBase):

    # @property 让 name 可以像普通属性一样访问：
    #
    # tool.name
    #
    # 而不是：
    #
    # tool.name()
    @property
    def name(self) -> str:

        # 这是 ToolRegistry 中使用的唯一工具名称。
        return "write_file"

    # description 是这个工具的文字说明。
    #
    # 后面 Prompt Builder 会把它告诉 LLM，
    # 让模型知道这个工具是干什么的。
    @property
    def description(self) -> str:

        # 告诉 LLM：
        #
        # 这个工具需要两个参数：
        #
        # path
        # content
        return (
            "向 workspace 中指定文件写入文本内容。"
            "参数：path 表示相对于 workspace 的文件路径，"
            "content 表示要写入的完整文本内容。"
        )

    # run() 是真正执行写文件操作的方法。
    def run(
        self,

        # arguments 是模型调用工具时提供的参数。
        #
        # 例如：
        #
        # {
        #     "path": "calculator.py",
        #     "content": "def add(a, b):\n    return a + b\n"
        # }
        arguments: dict[str, Any],

    # 所有 Tool 都统一返回 Observation。
    ) -> Observation:

        # get() 从 arguments 字典中读取 path。
        #
        # 如果不存在 path，
        # relative_path 就会得到 None。
        relative_path = arguments.get("path")

        # 获取模型希望写入文件的文本内容。
        #
        # 如果不存在 content，
        # content 就会得到 None。
        content = arguments.get("content")

        # 检查 path 是否是字符串。
        #
        # 如果模型漏掉 path，
        # 或者传入了错误类型，
        # 就直接返回失败结果。
        if not isinstance(relative_path, str):

            # 返回统一的 Observation。
            return Observation(
                content="write_file 缺少有效的 path 参数",
                success=False,
            )

        # 检查 content 是否是字符串。
        #
        # 即使我们想写入一个空文件：
        #
        # content = ""
        #
        # 也是合法字符串，因此允许。
        if not isinstance(content, str):

            # 参数错误时返回失败结果。
            return Observation(
                content="write_file 缺少有效的 content 参数",
                success=False,
            )

        # try / except 用来捕获文件操作过程中
        # 可能出现的异常。
        try:

            # 使用 FileToolBase 中已经实现的 _resolve_path()。
            #
            # 它会：
            #
            # 1. 拼接 workspace 和 relative_path
            # 2. 转成规范绝对路径
            # 3. 检查是否试图逃出 workspace
            target_path = self._resolve_path(relative_path)

            # 在真正写入之前，
            # 先记录这个文件原本是否存在。
            #
            # 后面的 metadata 可以利用这个信息判断：
            #
            # 是修改了旧文件，
            # 还是创建了新文件。
            existed_before = target_path.exists()

            # 如果目标路径已经存在，
            # 但它是一个文件夹而不是普通文件，
            # 就不能直接写入。
            if target_path.exists() and not target_path.is_file():

                # 返回失败信息。
                return Observation(
                    content=f"目标不是文件：{relative_path}",
                    success=False,
                )

            # target_path.parent 表示目标文件所在的父目录。
            #
            # 例如：
            #
            # target_path:
            # workspace/project/utils/math.py
            #
            # parent:
            # workspace/project/utils
            parent_directory = target_path.parent

            # mkdir() 用来创建目录。
            #
            # parents=True：
            # 如果中间目录不存在，也一起创建。
            #
            # exist_ok=True：
            # 如果目录已经存在，不要报错。
            #
            # 因此 Agent 可以直接写：
            #
            # src/utils/math.py
            #
            # 即使 src/utils 还不存在，也能够创建。
            parent_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            # write_text() 是 pathlib.Path 提供的写文本方法。
            #
            # 它的行为是：
            #
            # 文件不存在 → 创建文件
            #
            # 文件已存在 → 覆盖原来的全部内容
            #
            # encoding="utf-8"
            # 表示使用 UTF-8 编码保存文本。
            target_path.write_text(
                content,
                encoding="utf-8",
            )

            # 写入成功后返回 Observation。
            return Observation(

                # 给 Agent 一个简单明确的结果。
                content=f"文件写入成功：{relative_path}",

                # 表示 Tool 正常执行。
                success=True,

                # metadata 保存额外信息，
                # 方便后面调试、记录 trajectory。
                metadata={

                    # 保存相对于 workspace 的路径。
                    "path": relative_path,

                    # existed_before=True：
                    # 说明修改的是已有文件。
                    #
                    # False：
                    # 说明这次创建了新文件。
                    "existed_before": existed_before,
                },
            )

        # Exception 是 Python 中大多数普通异常的共同父类。
        #
        # 文件权限、路径错误等问题都可能进入这里。
        except Exception as error:

            # str(error) 把异常对象转换成文字，
            # 方便 Agent 看到失败原因。
            return Observation(
                content=f"写入文件失败：{str(error)}",
                success=False,
            )



# SearchTextTool 用来在 workspace 中搜索指定文本。
#
# SWE Agent 经常需要先定位：
#
# 某个函数在哪里
# 某个变量在哪里
# 某个错误字符串在哪里
#
# 所以搜索代码是非常基础的能力。
class SearchTextTool(FileToolBase):

    # 定义工具名称。
    @property
    def name(self) -> str:

        # LLM 后面会通过这个名字调用工具。
        return "search_text"

    # 定义工具说明。
    @property
    def description(self) -> str:

        # Prompt Builder 后面会把这段说明提供给 LLM。
        return (
            "在 workspace 的文件中搜索指定文本。"
            "参数：query 表示要搜索的文本；"
            "path 表示搜索起点，相对于 workspace，默认搜索整个 workspace。"
        )

    # run() 执行真正的文本搜索。
    def run(
        self,

        # arguments 是 LLM 提供给工具的参数。
        #
        # 例如：
        #
        # {
        #     "query": "def add",
        #     "path": "calculator"
        # }
        arguments: dict[str, Any],

    # 所有 Tool 都统一返回 Observation。
    ) -> Observation:

        # 从参数中取得需要搜索的文本。
        query = arguments.get("query")

        # 获取搜索起点。
        #
        # 如果 LLM 没有提供 path，
        # 默认使用 "."。
        #
        # "." 表示 workspace 本身。
        relative_path = arguments.get(
            "path",
            ".",
        )

        # 检查 query 是否是有效字符串。
        if not isinstance(query, str) or not query:

            # 参数错误时返回失败 Observation。
            return Observation(
                content="search_text 缺少有效的 query 参数",
                success=False,
            )

        # 检查 path 是否是字符串。
        if not isinstance(relative_path, str):

            # path 类型错误时返回失败。
            return Observation(
                content="search_text 缺少有效的 path 参数",
                success=False,
            )

        # 使用 try / except 捕获搜索过程中的异常。
        try:

            # 使用 FileToolBase 的安全路径解析方法。
            #
            # 这样 SearchTextTool 同样不能逃出 workspace。
            search_path = self._resolve_path(
                relative_path
            )

            # 检查搜索路径是否存在。
            if not search_path.exists():

                # 路径不存在时返回失败。
                return Observation(
                    content=f"搜索路径不存在：{relative_path}",
                    success=False,
                )

            # matches 用来保存所有搜索结果。
            #
            # 每一项以后类似：
            #
            # calculator.py:2:    return a + b
            matches: list[str] = []

            # 如果 search_path 本身就是一个文件，
            # 那么只搜索这一个文件。
            if search_path.is_file():

                # 放入列表，
                # 后面统一处理。
                files = [search_path]

            # 如果 search_path 是目录。
            else:

                # rglob("*") 是 pathlib.Path 提供的方法。
                #
                # 它会递归遍历当前目录以及所有子目录。
                #
                # 例如：
                #
                # workspace/
                # ├── main.py
                # └── utils/
                #     └── math.py
                #
                # 两个文件都会被找到。
                files = [
                    path
                    for path in search_path.rglob("*")
                    if path.is_file()
                ]

            # 依次检查每一个文件。
            for file_path in files:

                # 某些文件可能不是 UTF-8 文本文件，
                # 例如图片、数据库、二进制文件等。
                #
                # 因此单独使用 try / except。
                try:

                    # 使用 UTF-8 读取文件。
                    file_content = file_path.read_text(
                        encoding="utf-8"
                    )

                # UnicodeDecodeError 表示：
                #
                # 这个文件不能按照 UTF-8 文本正常解析。
                #
                # 对这种文件直接跳过即可。
                except UnicodeDecodeError:

                    # continue 表示：
                    # 跳过当前文件，继续处理下一个文件。
                    continue

                # splitlines() 把整个文件拆成一行一行。
                #
                # 例如：
                #
                # "a\nb\nc"
                #
                # 会得到：
                #
                # ["a", "b", "c"]
                lines = file_content.splitlines()

                # enumerate() 可以同时得到：
                #
                # 行号
                # 行内容
                #
                # start=1 表示行号从 1 开始，
                # 与编辑器里的行号保持一致。
                for line_number, line in enumerate(
                    lines,
                    start=1,
                ):

                    # 如果这一行包含 query，
                    # 就认为搜索命中。
                    if query in line:

                        # relative_to(self.workspace)
                        # 把完整绝对路径转换成
                        # 相对于 workspace 的路径。
                        relative_file = file_path.relative_to(
                            self.workspace
                        )

                        # 保存搜索结果。
                        #
                        # 例如：
                        #
                        # calculator.py:2:    return a + b
                        matches.append(
                            f"{relative_file}:{line_number}:{line}"
                        )

            # 如果一个匹配结果都没有。
            if not matches:

                # 注意：
                #
                # “没有搜到”并不代表 Tool 执行失败。
                #
                # 搜索本身是正常完成的，
                # 只是结果数量为 0。
                return Observation(
                    content=f"未找到文本：{query}",
                    success=True,

                    # metadata 保存结构化信息。
                    metadata={
                        "query": query,
                        "path": relative_path,
                        "match_count": 0,
                    },
                )

            # 使用 "\n".join(...)
            # 把所有匹配结果用换行连接起来。
            result = "\n".join(matches)

            # 搜索成功并且找到了结果。
            return Observation(
                content=result,
                success=True,

                # 保存一些额外信息。
                metadata={
                    "query": query,
                    "path": relative_path,
                    "match_count": len(matches),
                },
            )

        # 捕获其他异常。
        except Exception as error:

            # 把异常信息返回给 Agent。
            return Observation(
                content=f"搜索文本失败：{str(error)}",
                success=False,
            )

      
        
# ReplaceTextTool 用来修改文件中的指定文本。
#
# 与 WriteFileTool 不同：
# WriteFileTool 会覆盖整个文件。
# ReplaceTextTool 只替换指定的文本片段。
#
# 它继承 FileToolBase，因此可以直接使用：
# self.workspace
# self._resolve_path()
class ReplaceTextTool(FileToolBase):

    # 定义工具名称。
    # ToolRegistry 后面通过这个名称查找工具。
    @property
    def name(self) -> str:

        # 返回工具的唯一名称。
        return "replace_text"

    # 定义工具说明。
    # 后面的 PromptBuilder 会把它提供给 LLM。
    @property
    def description(self) -> str:

        return (
            "替换 workspace 中指定文件的一段文本。"
            "参数：path 为相对文件路径，"
            "old_text 为需要替换的原文本，"
            "new_text 为替换后的文本。"
            "只有原文本恰好出现一次时才允许替换。"
        )

    # 实现工具的核心执行方法。
    def run(
        self,
        arguments: dict[str, Any],
    ) -> Observation:

        # 获取需要修改的文件路径。
        # 如果没有提供 path，则返回 None。
        relative_path = arguments.get("path")

        # 获取需要被替换的旧文本。
        old_text = arguments.get("old_text")

        # 获取替换后的新文本。
        new_text = arguments.get("new_text")

        # 第一步：检查 path 参数。
        #
        # path 必须是非空字符串。
        if not isinstance(relative_path, str) or not relative_path.strip():

            # 参数错误，返回失败 Observation。
            return Observation(
                content="replace_text 缺少有效的 path 参数",
                success=False,
            )

        # 第二步：检查 old_text 参数。
        #
        # 旧文本必须是非空字符串。
        #
        # 如果 old_text=""，
        # Python 的字符串替换可能会在多个位置插入新文本，
        # 因此我们明确禁止空字符串。
        if not isinstance(old_text, str) or not old_text:

            # 返回参数错误信息。
            return Observation(
                content="replace_text 缺少有效的 old_text 参数",
                success=False,
            )

        # 第三步：检查 new_text 参数。
        #
        # new_text 允许是空字符串。
        # 因为空字符串可以用来删除某一段指定文本。
        if not isinstance(new_text, str):

            # 返回参数错误信息。
            return Observation(
                content="replace_text 缺少有效的 new_text 参数",
                success=False,
            )

        # 使用 try/except 捕获文件操作异常。
        try:

            # 使用 FileToolBase 提供的安全路径解析方法。
            #
            # _resolve_path() 会：
            # 1. 将相对路径转换为绝对路径
            # 2. 检查路径是否位于 workspace 内部
            # 3. 越界时抛出 ValueError
            target_path = self._resolve_path(relative_path)

            # 检查目标文件是否存在。
            if not target_path.exists():

                # 文件不存在时不能执行替换。
                return Observation(
                    content=f"文件不存在：{relative_path}",
                    success=False,
                )

            # 检查目标是否为普通文件。
            #
            # 防止错误地对文件夹进行文本操作。
            if not target_path.is_file():

                # 返回失败结果。
                return Observation(
                    content=f"目标不是文件：{relative_path}",
                    success=False,
                )

            # 第四步：读取原始文件。
            #
            # read_bytes() 是 pathlib.Path 提供的方法。
            # 它会读取文件的原始字节内容。
            #
            # decode("utf-8") 将字节转换为字符串。
            #
            # 这里没有使用 read_text()，
            # 是为了尽量保留原文件的换行符格式，
            # 避免在 Windows 中意外改变整份文件的换行方式。
            original_content = target_path.read_bytes().decode("utf-8")

            # 第五步：统计旧文本出现了多少次。
            #
            # str.count() 是 Python 字符串的方法。
            #
            # 例如：
            # "hello hello".count("hello")
            #
            # 结果为 2。
            match_count = original_content.count(old_text)

            # 如果一次也没有找到。
            if match_count == 0:

                # 不修改任何文件。
                return Observation(
                    content=f"没有找到需要替换的文本：{old_text}",
                    success=False,
                )

            # 如果旧文本出现多次。
            if match_count > 1:

                # 不允许直接修改。
                #
                # 这样可以防止 Agent 修改多个相似函数。
                return Observation(
                    content=(
                        f"旧文本出现了 {match_count} 次，"
                        "为了避免误修改，请提供更精确的 old_text。"
                    ),
                    success=False,
                )

            # 第六步：执行字符串替换。
            #
            # str.replace(old, new, count)
            #
            # 参数含义：
            # old   = 需要被替换的文本
            # new   = 替换后的文本
            # count = 最多替换多少次
            #
            # 这里指定 1，表示只替换一次。
            new_content = original_content.replace(
                old_text,
                new_text,
                1,
            )

            # 第七步：将新内容写回原文件。
            #
            # encode("utf-8") 将字符串转换成字节。
            #
            # write_bytes() 会将字节写入文件。
            # 如果文件存在，就覆盖原文件内容。
            #
            # 虽然这里写回了完整文件，
            # 但新内容只改变了指定的那一处文本。
            target_path.write_bytes(
                new_content.encode("utf-8")
            )

            # 第八步：返回成功的 Observation。
            return Observation(

                # 告诉 Agent 哪个文件修改成功。
                content=f"文本替换成功：{relative_path}",

                # 表示工具执行成功。
                success=True,

                # metadata 保存结构化信息。
                # 方便后续记录和分析 Agent 的修改行为。
                metadata={

                    # 保存被修改文件的相对路径。
                    "path": relative_path,

                    # 记录成功替换的次数。
                    "replacement_count": 1,

                    # 保存替换前的文本。
                    "old_text": old_text,

                    # 保存替换后的文本。
                    "new_text": new_text,
                },
            )

        # 捕获路径越界、文件权限或编码等异常。
        except Exception as error:

            # 将异常转换成 Observation。
            # 避免整个 Agent 因为文件错误直接崩溃。
            return Observation(
                content=f"文本替换失败：{str(error)}",
                success=False,
            )