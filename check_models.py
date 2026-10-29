# os 用于读取环境变量。
import os

# load_dotenv 用于加载 .env 配置。
from dotenv import load_dotenv

# OpenAI 是我们使用的 API 客户端。
from openai import OpenAI


# 加载项目中的 .env 文件。
load_dotenv()

# 创建当前第三方平台的 API 客户端。
client = OpenAI(
    # 从环境变量读取密钥。
    api_key=os.environ["LLM_API_KEY"],

    # 使用当前配置的 API 基础地址。
    base_url=os.getenv("LLM_BASE_URL"),

    # 调试时设置 15 秒超时。
    timeout=15.0,

    # 暂时关闭自动重试。
    max_retries=0,
)

# 开始查询。
print("正在查询可用模型...", flush=True)

# 捕获网络错误或接口错误。
try:
    # models.list() 会请求 GET /v1/models。
    # 这是一种常见的 OpenAI 兼容模型查询接口。
    models = client.models.list()

    # 遍历服务器返回的模型。
    for model in models.data:

        # 只显示名称包含 deepseek 的模型。
        if "deepseek" in model.id.lower():

            # 输出可以用于 API 调用的模型 ID。
            print(model.id)

# 查询失败时输出错误。
except Exception as error:

    # 显示异常类型和原因。
    print(f"查询失败：{type(error).__name__}: {error}")