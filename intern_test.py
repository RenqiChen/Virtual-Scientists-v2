import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-gaqeyxvrpmjondtmihuydxjevoztjgibkfmfqyhgmonhfsml", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1",
    timeout=60,
    max_retries=3,
)
response = client.chat.completions.create(
    model="OpenGVLab/InternVL2-26B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png"
                    }
                }
            ]
        }],
    stream=False
)

for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end='', flush=True)