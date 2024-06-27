from dotenv import load_dotenv

# AsyncIteratorCallbackHandler 和其他相关导入用于处理异步流式输出。
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
# asyncio 是Python的异步I/O框架，用于编写并发代码。
import asyncio
# sys 用于标准输入输出操作。
import sys
from dotenv import load_dotenv
import os
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_community.llms.tongyi import Tongyi
# 最新的导入方式，下面两种都废弃了
from langchain_openai import ChatOpenAI

# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI

load_dotenv(override=True)

# 设置回调处理器: AsyncIteratorCallbackHandler是一个异步回调处理器，它允许模型的响应以迭代的方式被处理，这对于流式输出特别有用，
# 比如逐字显示生成的文本而不是等待完整响应。
handler = AsyncIteratorCallbackHandler()

# CloudflareWorkersAI
cf_llm = CloudflareWorkersAI(
    account_id=os.getenv('CF_ACCOUNT_ID'),
    api_token=os.getenv('CF_API_TOKEN'),
    model='@cf/meta/llama-3-8b-instruct',
    streaming=True,
    callbacks=[handler]
)

# qwen
qwen_llm = Tongyi(
    model='qwen2-1.5b-instruct',
    streaming=True,
    callbacks=[handler]
)

# openai/moonshot
ms_llm = ChatOpenAI(
    openai_api_base=os.getenv('OPENAI_API_BASE'),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="moonshot-v1-8k",
    temperature=0.7,
    streaming=True,
    callbacks=[handler]
)

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
qw_llm_openai = ChatOpenAI(
    openai_api_base=os.getenv('DASHSCOPE_API_BASE'),
    openai_api_key=DASHSCOPE_API_KEY,
    model_name="qwen2-1.5b-instruct",
    temperature=0.7,
    streaming=True,
)


# llm = ChatOpenAI(streaming=True, callbacks=[handler], temperature=0)


async def consumer():
    iterator = handler.aiter()
    async for item in iterator:
        sys.stdout.write(item)
        sys.stdout.flush()


# 这段代码是一个使用Python编写的异步程序，其主要目的是与OpenAI的Chat模型交互并获取流式响应
# 其他模型 的流式输出（LLM）
if __name__ == '__main__':
    message = "What is AI?"
    loop = asyncio.get_event_loop()
    loop.create_task(qwen_llm.agenerate(prompts=[message]))
    loop.create_task(consumer())
    loop.run_forever()
    loop.close()

# openai 的流式输出（chat-model）
if __name__ == '__main__':
    message = "你是谁啊?"
    loop = asyncio.get_event_loop()
    loop.create_task(ms_llm.agenerate(messages=[[HumanMessage(content=message)]]))
    loop.create_task(consumer())
    loop.run_forever()
    loop.close()

if __name__ == '__main__':
    message = "你是什么大模型?"
    loop = asyncio.get_event_loop()
    loop.create_task(qw_llm_openai.agenerate(messages=[[HumanMessage(content=message)]]))
    loop.create_task(consumer())
    loop.run_forever()
    loop.close()
