# json 是 Python 标准库。
#
# JSON 是 LLM 与我们的程序之间
# 非常适合使用的一种结构化数据格式。
#
# json.loads() 可以把 JSON 字符串
# 转换成 Python 对象。
import json


# Any 表示任意数据类型。
#
# JSON 解析之后可能得到：
# dict、list、str、int 等类型。
from typing import Any


# 导入我们之前定义好的 Action。
#
# Parser 最终的工作就是：
#
# 字符串
#   ↓
# Action
from swe_agent.agent.state import Action


# ActionParser 专门负责解析 LLM 的输出。
class ActionParser:

    # parse() 接收模型输出的字符串，
    # 最终返回一个标准 Action 对象。
    def parse(
        self,
        text: str,
    ) -> Action:

        # 首先检查输入是不是字符串。
        #
        # 正常情况下 LLM 返回的内容应该是 str。
        if not isinstance(text, str):

            # Parser 不是 Tool，
            # 因此这里不返回 Observation，
            # 而是直接抛出 ValueError。
            raise ValueError(
                "LLM 输出必须是字符串"
            )

        # strip() 去掉字符串开头和结尾的：
        #
        # 空格
        # 换行
        # 制表符
        cleaned_text = text.strip()

        # 有些 LLM 即使我们要求只输出 JSON，
        # 仍然可能返回：
        #
        # ```json
        # {...}
        # ```
        #
        # 所以这里允许最常见的 Markdown 代码块格式。
        if cleaned_text.startswith("```"):

            # splitlines() 把文本按照行拆开。
            lines = cleaned_text.splitlines()

            # 如果第一行是：
            #
            # ```json
            #
            # 或：
            #
            # ```
            #
            # 就删除第一行。
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            # 如果最后一行是：
            #
            # ```
            #
            # 就删除最后一行。
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            # 再重新组合成字符串。
            cleaned_text = "\n".join(lines).strip()

        # 尝试把 JSON 字符串解析成 Python 对象。
        try:

            # json.loads()：
            #
            # JSON 字符串
            #     ↓
            # Python 对象
            #
            # 例如：
            #
            # '{"tool": "read_file"}'
            #
            # 会变成：
            #
            # {"tool": "read_file"}
            data: Any = json.loads(cleaned_text)

        # JSONDecodeError 表示：
        #
        # 字符串不是合法 JSON。
        except json.JSONDecodeError as error:

            # 把底层 JSON 错误包装成更容易理解的信息。
            raise ValueError(
                f"LLM 输出不是合法 JSON：{str(error)}"
            ) from error

        # 一个 Action 必须是 JSON 对象。
        #
        # JSON：
        #
        # {
        #     "tool": "...",
        #     "arguments": {}
        # }
        #
        # 解析到 Python 后应该是 dict。
        if not isinstance(data, dict):

            raise ValueError(
                "LLM 输出必须是一个 JSON 对象"
            )

        # 从 JSON 中获取 tool。
        tool = data.get("tool")

        # tool 必须是非空字符串。
        if not isinstance(tool, str) or not tool.strip():

            raise ValueError(
                "Action 缺少有效的 tool 字段"
            )

        # 获取 arguments。
        #
        # 如果模型没有提供 arguments，
        # 默认使用空字典。
        arguments = data.get(
            "arguments",
            {},
        )

        # arguments 必须是 JSON 对象，
        # 也就是 Python 中的 dict。
        if not isinstance(arguments, dict):

            raise ValueError(
                "Action 的 arguments 必须是 JSON 对象"
            )

        # 获取 reasoning。
        #
        # reasoning 是模型给出的简短决策说明，
        # 目前允许不提供。
        reasoning = data.get(
            "reasoning",
            "",
        )

        # reasoning 如果存在，
        # 必须是字符串。
        if not isinstance(reasoning, str):

            raise ValueError(
                "Action 的 reasoning 必须是字符串"
            )

        # 所有字段检查通过以后，
        # 创建我们阶段 2 定义的 Action 对象。
        action = Action(

            # strip() 去掉工具名前后的多余空格。
            tool=tool.strip(),

            # 保存工具参数。
            arguments=arguments,

            # 保存简短决策说明。
            reasoning=reasoning,
        )

        # 返回标准化后的 Action。
        return action