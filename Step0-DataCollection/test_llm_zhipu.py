# 安装最新版本
# pip install zai-sdk

import os

from zai import ZhipuAiClient
client = ZhipuAiClient(api_key=os.getenv("ZHIPUAI_API_KEY"))
response = client.chat.completions.create(
        model='glm-4.7',
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is the revolution of llm?"}
        ],
        stream=True,
        thinking={
            "type": "enabled"
        }
    )
for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end='')
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
