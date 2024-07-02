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
