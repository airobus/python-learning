import json
from fastapi.responses import FileResponse
from fastapi import FastAPI
# from config import SUPABASE
from pathlib import Path
import logging
from robus.config import SUPABASE, ms_llm, cf_llm, qw_llm, qw_llm_openai, conversationChain, \
    vectorstore, embeddings, subabase_retriever
from fastapi.responses import Response, StreamingResponse, JSONResponse
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, \
    ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains.conversation.base import ConversationChain
from typing import AsyncIterable, Awaitable, Iterator
import asyncio
from typing import Dict, Any, Iterator
from starlette.schemas import OpenAPIResponse
from langchain_core.messages import BaseMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableConfig
from langchain.load.serializable import Serializable
from typing import Optional, Dict
from langchain_core.runnables.utils import Input
from langchain.schema.runnable import Runnable
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate

# >>>>>>>>>>基础>>>>>>>>>>>>>>
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


# 基于: rag + memory 的问答 ❌
@app.post("/rag/memory/ask")
def ask(body: dict):
    question = body['question']

    chat_prompt = ChatPromptTemplate.from_template(prompt())
    chat_history = conversationChain.memory.buffer
    partial_prompt = chat_prompt.partial(chat_history=chat_history)

    prompt_chain = chat_prompt | qw_llm_openai

    rag_chain = (
            {"context": (subabase_retriever | format_docs), "question": RunnablePassthrough()}
            | StdOutputRunnable
            | qw_llm_openai
            | StrOutputParser()
    )

    print(rag_chain)

    result = rag_chain.invoke(question)

    return result


# stream式暂时不可行 ❌
@app.post("/stream/rag/ask")
def ask(body: dict):
    question = body['question']

    rag_chain = (
            {"context": (subabase_retriever | format_docs), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt())
            | qw_llm_openai
            | StrOutputParser()
    )

    result = rag_chain.stream(question)

    def predict():
        text = ""
        for _token in result:
            js_data = {"code": "200", "msg": "ok", "data": _token}
            yield f"data: {json.dumps(js_data, ensure_ascii=False)}\n\n"
            text += _token
        print(text)

    generate = predict()
    return StreamingResponse(generate, media_type="text/event-stream")


# 基于知识库的问答
@app.post("/rag/ask")
def ask(body: dict):
    question = body['question']

    rag_chain = (
            {"context": (subabase_retriever | format_docs), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(prompt())
            | qw_llm_openai
            | StrOutputParser()
    )

    result = rag_chain.invoke(question)

    return result


# 暂时不是stream输出方式 ❌
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


# >>>>>>>>>>类>>>>>>>>>>>>>>>
# 自定义一个继承Runnable的类
class StdOutputRunnable(Serializable, Runnable[Input, Input]):
    @property
    def lc_serializable(self) -> bool:
        return True

    def invoke(self, input: Dict, config: Optional[RunnableConfig] = None) -> Input:
        print(f"Hey, I received the name {input['name']}")
        return self._call_with_config(lambda x: x, input, config)


# >>>>>>>>>>方法>>>>>>>>>>>>>>>
def call_llm(question: str):
    return cf_llm.invoke(question)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def prompt():
    return '''
    # Character
    You're a knowledgeable assistant capable of providing concise answers to a variety of questions, drawing from the context provided, and admitting when you don't know the answer.
    
    ## Skills
    1. **Answering Questions:** Utilize the given context to answer user questions. If the answer is not clear from the context, truthfully state that the answer is unknown to maintain accuracy in your responses.
    Question: {question}
    Context: {context}    
    2. You are a nice chatbot having a conversation with a human.
    Previous conversation:
    {chat_history}
    
    ### Answering Questions Format:
    - Answer:  
    
    ## Constraints:
    - Keep answers to a maximum of three sentences to maintain brevity.
    - If the answer cannot be determined, simply confess that you do not know. Honesty is paramount in maintaining credibility.
    - If the answer is not reflected in the context, please reply: Sorry, I don't know for the moment.
    - Focus on gleaning answers from the context provided only.
    - All questions should be answered in Chinese
    '''


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
