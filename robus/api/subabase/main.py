import json
from fastapi.responses import FileResponse
from fastapi import FastAPI
# from config import SUPABASE
from pathlib import Path
import logging
from robus.config import SUPABASE, ms_llm, cf_llm, qw_llm, qw_llm_openai, conversationChain
from fastapi.responses import Response, StreamingResponse, JSONResponse
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, \
    ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains.conversation.base import ConversationChain
from typing import AsyncIterable, Awaitable, Iterator
import asyncio
from typing import Dict, Any, Iterator
from starlette.schemas import OpenAPIResponse
from langchain_core.messages import BaseMessageChunk

# >>>>>>>>>>基础>>>>>>>>>>>>
log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()


# >>>>>>>>>>接口>>>>>>>>>>>>>>>
# 返回页面
@app.get("/")
async def subabase_homepage():
    file_path = Path(__file__).parent / "index.html"
    return FileResponse(file_path)


@app.post("/ask")
def ask(body: dict):
    return Response(call_llm(body['question']))


# 基于知识库的问答
@app.post("/rag/ask")
def ask(body: dict):
    result = qw_llm_openai.stream(body['question'])

    # Streaming 返回
    return StreamingResponse(
        (str(chunk.content) for chunk in result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


# 暂时不是stream输出方式
# stream式 + memory，输出
@app.post("/stream/memory/ask")
def ask(body: dict):
    result = conversationChain.stream(body['question'])

    def generate():
        for output in result:
            # 确保输出是字符串
            # yield str(output).encode('utf-8') + b'\n'
            yield f"data: {str(output)}\n\n".encode('utf-8')

    return StreamingResponse(generate(), media_type="text/event-stream")


# 存在记忆 的输出
@app.post("/memory/ask")
def ask(body: dict):
    result = conversationChain.invoke(body['question'])
    # result = {'input': '给我一片作文  ', 'history': '', 'response': '好的，我可以帮您写一篇作文。请问您需要什么样的主题或者内容？'}
    print(result)
    return JSONResponse(result)


# stream式输出
@app.post("/stream/ask")
def ask(body: dict):
    result = qw_llm_openai.stream(body['question'])

    # Streaming 返回
    return StreamingResponse(
        (str(chunk.content) for chunk in result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/chat")
async def generate_chat_completion():
    r = call_llm("chat")
    response = StreamingResponse(
        r.content,
        status_code=r.status,
        headers=dict(r.headers),
    )
    content_type = response.headers.get("Content-Type")
    citations = []
    if "text/event-stream" in content_type:
        return StreamingResponse(
            openai_stream_wrapper(response.body_iterator, citations),
        )
    if "application/x-ndjson" in content_type:
        return StreamingResponse(
            ollama_stream_wrapper(response.body_iterator, citations),
        )


@app.get("/select")
def select():
    response = SUPABASE.table('documents').select("*").execute()
    return response


# >>>>>>>>>>方法>>>>>>>>>>>>>>>
def call_llm(question: str):
    return cf_llm.invoke(question)


# 这段代码定义了一个异步生成器函数 openai_stream_wrapper，它包装了另一个异步生成器 original_generator。
# 这个函数的主要目的是在原始生成器产生的数据流前面添加一些特定的信息，然后继续生成原始数据。
# citations 可能是需要在数据流开始时发送的一些引用或元数据。
# {"citations": ["citation1", "citation2"]}
async def openai_stream_wrapper(original_generator, citations):
    # 第一次yield返回这个
    yield f"data: {json.dumps({'citations': citations})}\n\n"
    # 然后返回这个数据
    async for data in original_generator:
        yield data


# {"citations": ["citation1", "citation2"]}
async def ollama_stream_wrapper(original_generator, citations):
    # 第一次yield返回这个
    yield f"{json.dumps({'citations': citations})}\n"
    # 然后返回这个数据
    async for data in original_generator:
        yield data


async def wait_done(fn: Awaitable, event: asyncio.Event):
    try:
        await fn
    except Exception as e:
        print(e)
        event.set()
    finally:
        event.set()


def yield_json(obj: Dict[str, Any]) -> Iterator[bytes]:
    """Yield a JSON object as a bytestring."""
    yield f"data: {json.dumps(obj)}\n\n".encode()
