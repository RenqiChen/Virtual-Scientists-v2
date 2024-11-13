import requests
import ollama
import os
from openai import OpenAI, Stream
import numpy as np
# os.environ["OLLAMA_API_URL"] = "http://paraai-n32-h-01-agent-152:11434/v1"
# print(os.environ["OLLAMA_API_URL"])
# # os.environ["OLLAMA_API_URL"] = "http://paraai-n32-h-01-agent-188:11434/v1"
# # 替换为实际的 IP 地址和端口
# url = "http://paraai-n32-h-01-agent-218:11440/v1"
# # url = "http://127.0.0.1:11434/v1"
# key_string = "damn"
# # response = ollama.chat(model="llama3.1", messages=[{
# #     "role":"user", 
# #     "content": "What is the name of the album with the most tracks?"
# # }])
# query_vector = ollama.embeddings(model="mxbai-embed-large", prompt=key_string)
# print(query_vector)
# _url = "http://127.0.0.1:11450/v1"
# model_config_dict={}
# _client = OpenAI(
#     timeout=60,
#     max_retries=3,
#     base_url=_url,
#     api_key="ollama",  # required but ignored
# )
# messages = "What is the name of the album with the most tracks?"
# model_type = "mxbai-embed-large"
# response = _client.embeddings.create(
#     input=messages,
#     model=model_type,
#     **model_config_dict,
# )
# print(np.array([response.data[0].embedding]).shape)

min_sleep_time = 0.1
max_sleep_time = 0.4

sleep_time = min_sleep_time + max_sleep_time*10/48

print(sleep_time)