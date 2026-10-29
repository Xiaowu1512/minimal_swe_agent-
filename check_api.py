# os 是 Python 标准库，用于读取环境变量。
import os

# time 是 Python 标准库，用于计算 API 请求耗时。
import time

# Path 用于定位项目根目录下的 .env 文件。
from pathlib import Path

# load_dotenv() 用于加载 .env 中的配置信息。
from dotenv import load_dotenv

# OpenAI 是我们之前使用的 API 客户端。
from openai import OpenAI


# 读取当前脚本所在目录下的 .env 文件。
load_dotenv(Path(__file__).resolve().parent / ".env")

# 读取 API Key。
api_key = os.getenv("LLM_API_KEY")

# 读取 API 基础地址。
base_url = os.getenv(
    "LLM_BASE_URL",
    "https://api.shubiaobiao.cn/v1"
)

# 读取模型名称。
model = os.getenv("LLM_MODEL", "deepseek-v3")

# 检查是否配置了 API Key。
if not api_key:
    raise ValueError("没有找到 LLM_API_KEY，请检查 .env 文件")

# 创建 API 客户端。
client = OpenAI(
    api_key=api_key,      # 使用 .env 中的密钥。
    base_url=base_url,    # 使用之前配置的模型平台。
    timeout=20.0,        # 单次网络操作的超时时间为 20 秒。
    max_retries=0,       # 禁止自动重试，便于排查响应缓慢的问题。
)

# 打印当前使用的模型。
print(f"测试模型：{model}", flush=True)

# 打印请求开始信息。
print("正在请求 API...", flush=True)

# perf_counter() 用于精确测量经过的时间。
start_time = time.perf_counter()

# 使用 try 捕获 API 调用过程中可能出现的异常。
try:

    # 向 DeepSeek-V3 发送一次简单请求。
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "只回复：API连接成功"
            }
        ],
        max_tokens=50,  # 限制最大输出长度，减少测试开销。
    )

    # 提取模型返回的文本。
    result = response.choices[0].message.content

    # 打印实际响应。
    print(f"模型回复：{result}", flush=True)

# 如果调用失败，输出具体异常类型和内容。
except Exception as error:

    # 打印异常类型，例如 APITimeoutError。
    print(f"请求失败，错误类型：{type(error).__name__}")

    # 打印具体错误原因。
    print(f"错误信息：{error}")

# finally 无论成功还是失败都会执行。
finally:

    # 计算本次请求实际耗时。
    elapsed = time.perf_counter() - start_time

    # 打印耗时，保留两位小数。
    print(f"总耗时：{elapsed:.2f} 秒")