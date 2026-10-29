# os 是 Python 自带的标准库。
# 这里主要使用它读取环境变量，例如 LLM_API_KEY。
import os


# dotenv 来自刚刚安装的 python-dotenv。
# load_dotenv() 的作用是读取项目根目录中的 .env 文件，
# 并把其中的变量加载到当前 Python 程序的环境变量中。
from dotenv import load_dotenv


# OpenAI 是 openai Python SDK 提供的客户端类。
# 虽然我们现在调用的不是 OpenAI 官方服务器，
# 但这个平台提供了 OpenAI 兼容接口，因此仍然可以使用这个 SDK。
from openai import OpenAI


# 从我们刚刚编写的 base.py 中导入 BaseLLM。
# 前面的 "." 表示“当前 llm 包中的 base.py”。
from .base import BaseLLM


# 读取项目根目录中的 .env 文件。
# 执行后，LLM_API_KEY、LLM_BASE_URL 等变量就可以通过 os 读取。
load_dotenv()


# 定义鼠标标平台对应的 LLM 客户端。
# 它继承 BaseLLM，因此必须实现 generate() 方法。
class ShubiaobiaoLLM(BaseLLM):

    # __init__ 是 Python 类的初始化方法。
    # 当我们执行 ShubiaobiaoLLM() 时，它会自动执行。
    def __init__(self):

        # os.getenv() 用于读取环境变量。
        # 第一个参数是变量名称。
        # 第二个参数 "deepseek-v3" 是默认值。
        #
        # 也就是说：
        # 如果 .env 中配置了 LLM_MODEL，就使用配置值；
        # 如果没有配置，就默认使用 deepseek-v3。
        self.model = os.getenv(
            "LLM_MODEL",
            "deepseek-v3",
        )

        # 创建一个 OpenAI SDK 客户端。
        # self.client 以后专门负责向大模型服务器发送请求。
        self.client = OpenAI(

            # os.environ["LLM_API_KEY"]
            # 用来读取 .env 中的 API Key。
            #
            # 和 getenv 不同：
            # 如果 LLM_API_KEY 不存在，
            # os.environ[...] 会直接报错，
            # 可以帮助我们及时发现 API Key 没配置。
            api_key=os.environ["LLM_API_KEY"],

            # base_url 指定 API 服务器地址。
            #
            # 默认并不是访问 OpenAI 官方服务器，
            # 而是访问我们现在使用的大模型平台。
            base_url=os.getenv(
                "LLM_BASE_URL",
                "https://api.shubiaobiao.cn/v1",
            ),
        )

    # 实现 BaseLLM 中规定的 generate() 方法。
    #
    # 输入：
    # prompt，例如“请解释 Transformer”
    #
    # 输出：
    # LLM 返回的文本。
    def generate(self, prompt: str) -> str:

        # 调用 chat.completions.create() 向 LLM API 发送请求。
        #
        # chat.completions 是一种非常常见的
        # OpenAI 兼容聊天接口格式。
        response = self.client.chat.completions.create(

            # 指定要调用的模型。
            # 当前就是 deepseek-v3。
            model=self.model,

            # messages 表示发送给模型的对话内容。
            # 它是一个 Python list（列表）。
            messages=[

                # 每一条消息使用 dict（字典）表示。
                {
                    # role 表示消息是谁说的。
                    # "user" 就代表用户消息。
                    "role": "user",

                    # content 是真正发送给模型的文本。
                    "content": prompt,
                }
            ],
        )

        # response 是 API 返回的完整对象。
        #
        # choices[0]
        # 表示取模型返回的第一个答案。
        #
        # message
        # 表示这条回答消息。
        #
        # content
        # 就是模型真正生成的文字。
        content = response.choices[0].message.content

        # 正常情况下 content 是字符串。
        # 如果特殊情况下返回 None，则返回空字符串，
        # 防止后面的 Agent 因为 None 出错。
        return content or ""