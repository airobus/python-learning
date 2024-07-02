# 最新的Embedding方式
# cloudflare_workersai
from langchain_community.embeddings.cloudflare_workersai import (
    CloudflareWorkersAIEmbeddings,
)
import os
from dotenv import load_dotenv

load_dotenv()
account_id = os.getenv('CF_ACCOUNT_ID')
api_token = os.getenv('CF_API_TOKEN')

# @cf/baai/bge-large-en-v1.5
# 维度是：1024

# @cf/baai/bge-small-en-v1.5
# 维度是：384
embeddings = CloudflareWorkersAIEmbeddings(
    account_id=account_id,
    api_token=api_token,
    model_name="@cf/baai/bge-small-en-v1.5",
)
