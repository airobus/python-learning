import asyncio
import uvicorn
import os
from typing import AsyncIterable, Awaitable
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_community.llms.tongyi import Tongyi

load_dotenv(override=True)


async def wait_done(fn: Awaitable, event: asyncio.Event):
    try:
        await fn
    except Exception as e:
        print(e)
        event.set()
    finally:
        event.set()


async def call_llm(question: str, prompt: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()

    # model = ChatOpenAI(streaming=True, verbose=True, callbacks=[callback])
    # coroutine = wait_done(ms_llm.agenerate(messages=[[HumanMessage(content=question)]]), callback.done)

    # ms_llm = ChatOpenAI(
    #     openai_api_base=os.getenv('OPENAI_API_BASE'),
    #     openai_api_key=os.getenv('OPENAI_API_KEY'),
    #     model_name="moonshot-v1-8k",
    #     temperature=0.7,
    #     streaming=True,
    #     callbacks=[callback]
    # )

    # cf_llm = CloudflareWorkersAI(
    #     account_id=os.getenv('CF_ACCOUNT_ID'),
    #     api_token=os.getenv('CF_API_TOKEN'),
    #     model='@cf/meta/llama-3-8b-instruct',
    #     streaming=True,
    #     callbacks=[callback]
    # )

    qw_llm = Tongyi(
        model='qwen2-1.5b-instruct',
        streaming=True,
        callbacks=[callback]
    )
    prompts = [question]

    # if prompt:
    #     prompts.append(prompt)

    coroutine = wait_done(qw_llm.agenerate(prompts=prompts), callback.done)

    task = asyncio.create_task(coroutine)

    async for token in callback.aiter():
        yield f"{token}"

    await task


app = FastAPI()


@app.post("/ask")
def ask(body: dict):
    return StreamingResponse(call_llm(body['question'], body['prompt']), media_type="text/event-stream")


@app.get("/")
async def homepage():
    return FileResponse('statics/index.html')


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8888, app=app)
