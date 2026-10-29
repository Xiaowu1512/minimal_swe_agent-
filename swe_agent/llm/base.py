# 从 Python 标准库 abc 中导入 ABC 和 abstractmethod。
# abc = Abstract Base Class，即“抽象基类”。
# 它主要用来规定：子类必须实现哪些方法。
from abc import ABC, abstractmethod


# 定义所有大语言模型客户端的共同父类。
# ABC 表示这是一个“抽象类”，主要用来制定规则，
# 通常不会直接创建 BaseLLM() 对象。
class BaseLLM(ABC):

    # abstractmethod 表示下面这个方法是“抽象方法”。
    # 任何继承 BaseLLM 的子类，都必须实现 generate()。
    @abstractmethod
    def generate(self, prompt: str) -> str:
        # prompt: str
        # 表示输入参数 prompt 应该是字符串。

        # -> str
        # 表示这个函数最终应该返回一个字符串。

        # 这里不写具体实现，因为不同模型调用 API 的方式可能不同。
        pass