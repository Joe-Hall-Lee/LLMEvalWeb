from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的变量
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def call_model(prompt, modelname):
    if "qwen" in modelname.lower():
        BASE_URL = DASHSCOPE_BASE_URL
        API_KEY = os.getenv("DASHSCOPE_API_KEY")
    elif "deepseek" in modelname.lower():
        BASE_URL = ARK_BASE_URL
        API_KEY = os.getenv("ARK_API_KEY")
    else:
        print("模型名称不正确，请检查模型名称！")
        return
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

        completion = client.chat.completions.create(
            model=modelname,
            messages=prompt
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")
