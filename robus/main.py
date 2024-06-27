#!/usr/bin/env python3
from public.usage import USAGE as html
import logging
from api.hello import router as hello_router
from api.subabase.main import app as subabase_router
import asyncio
import uvicorn
from typing import AsyncIterable, Awaitable
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_community.llms.tongyi import Tongyi
from config import AppConfig, PersistentConfigTest

log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()

# 两种方式
app.include_router(hello_router, prefix="/hello")
app.mount("/test", hello_router)
# 核心
app.mount("/subabase", subabase_router)

load_dotenv(override=True)

# 在应用程序启动时执行的初始化操作，方式1
app.state.config = AppConfig()
app.state.config.PersistentConfigTest = PersistentConfigTest


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

    coroutine = wait_done(qw_llm.agenerate(prompts=prompts), callback.done)

    task = asyncio.create_task(coroutine)

    async for token in callback.aiter():
        yield f"{token}"

    await task


# @app.get("/")
# def _root():
#     return Response(content=html, media_type="text/html")


@app.post("/ask")
def ask(body: dict):
    return StreamingResponse(call_llm(body['question'], body['prompt']), media_type="text/event-stream")


@app.get("/")
async def homepage():
    return FileResponse('public/index.html')


# print(app.state.config.PersistentConfigTest)


# 在应用程序启动时执行的初始化操作，方式2
@app.on_event("startup")
async def startup_event():
    # 例如，连接到数据库、加载配置、初始化缓存等
    log.info(" log-Initializing data...")


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8888, app=app)
