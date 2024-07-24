#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langserve import add_routes
import os
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_community.llms.tongyi import Tongyi
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

account_id = os.getenv('CF_ACCOUNT_ID')
api_token = os.getenv('CF_API_TOKEN')
print(account_id)
print(api_token)

# CloudflareWorkersAI
model = '@cf/meta/llama-3-8b-instruct'
cf_llm = CloudflareWorkersAI(
    account_id=account_id,
    api_token=api_token,
    model=model
)

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
print(DASHSCOPE_API_KEY)

# qwen
qw_llm = Tongyi(
    model='qwen2-1.5b-instruct'
)

# qwen 兼容 openai的接口
qw_llm_openai = ChatOpenAI(
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    openai_api_key=DASHSCOPE_API_KEY,
    model_name="qwen2-1.5b-instruct",
    temperature=0.7,
    streaming=True,
)

api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')
print(api_key)
print(base_url)

# openai/moonshot
ms_llm = ChatOpenAI(
    openai_api_base=base_url,
    openai_api_key=api_key,
    model_name="moonshot-v1-8k",
    temperature=0.7,
)
# # # # # # # # # # # #
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# 在给定的 FastAPI 应用程序或 APIRouter 上注册路由。
add_routes(
    app,
    qw_llm_openai,
    path="/openai",
)

system_message_prompt = SystemMessagePromptTemplate.from_template("""
    You are secrets.toml helpful assistant that translates {input_language} to {output_language}.
""")
human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

add_routes(
    app,
    chat_prompt | qw_llm_openai,
    path="/translate",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9999)
