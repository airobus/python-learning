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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableConfig
from langchain.load.serializable import Serializable
from typing import Optional, Dict
from langchain_core.runnables.utils import Input
from langchain.schema.runnable import Runnable
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains.summarize import load_summarize_chain

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


# 返回页面2
@app.get("/upload")
async def subabase_homepage():
    file_path = Path(__file__).parent / "upload.html"
    return FileResponse(file_path)


@app.post("/ask")
def ask(body: dict):
    return Response(call_llm(body['question']))


@app.post("/add/url")
async def loader_url(body: dict):
    link = str(body['link']).strip()
    loader = UnstructuredURLLoader(urls=[link])
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    # 向量化存储
    SupabaseVectorStore.from_documents(
        splits,
        embeddings,
        client=SUPABASE,
        table_name="bge_small_vector",
        query_name="bge_small_match_documents",
    )

    return f'分割成{len(splits)}个文档'


# stream + rag + memory ❌ 暂时没有memory
@app.post("/stream/rag/memory/ask")
def ask(body: dict):
    question = body['question']

    chat_prompt = ChatPromptTemplate.from_template(prompt())
    chat_history = conversationChain.memory.buffer
    print(chat_history)

    # 预处理输入的数据
    def pre_input(prompt: str) -> Dict:
        # rag
        context = str(subabase_retriever | format_docs)
        return {
            "context": context,
            "question": prompt,
            "chat_history": chat_history
        }

    chain = (
            RunnableLambda(pre_input) |
            {
                "context": itemgetter('context'),
                "question": itemgetter('question'),
                "chat_history": itemgetter('chat_history'),
            }
            | chat_prompt
            | RunnablePassthrough()
            | StdOutputRunnable()
            | qw_llm_openai
            | StrOutputParser()
    )

    # 流式返回
    def generate():
        for chunk in chain.stream(question):
            for key in chunk:
                yield key

    return StreamingResponse(generate(), media_type="text/event-stream")


# 基于: rag + memory 的问答
@app.post("/rag/memory/ask")
def ask(body: dict):
    question = body['question']

    chat_prompt = ChatPromptTemplate.from_template(prompt())
    chat_history = conversationChain.memory.buffer

    # 预处理输入的数据
    def pre_input(prompt: str) -> Dict:
        context = str(subabase_retriever | format_docs)
        return {
            "context": context,
            "question": prompt,
            "chat_history": chat_history
        }

    rag_chain = (
            RunnableLambda(pre_input) |
            {
                "context": itemgetter('context'),
                "question": itemgetter('question'),
                "chat_history": itemgetter('chat_history')
            }
            | chat_prompt
            | RunnablePassthrough()
            | StdOutputRunnable()
            | qw_llm_openai
            | StrOutputParser()
    )

    # print(rag_chain)

    result = rag_chain.invoke(question)

    return result


# stream + rag
@app.post("/stream/rag/ask")
def ask(body: dict):
    question = body['question']

    rag_chain = (
            {"context": (subabase_retriever | format_docs), "question": (RunnablePassthrough() | StdOutputRunnable())}
            | ChatPromptTemplate.from_template(stream_rag_prompt())
            | qw_llm_openai
            | StrOutputParser()
    )

    # result = rag_chain.stream(question)
    def generate():
        for chunk in rag_chain.stream(question):
            for key in chunk:
                yield key

    return StreamingResponse(generate(), media_type="text/event-stream")


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
    question = body['question']

    def generate():
        for chunk in conversationChain.stream(question):
            for key in chunk:
                print(key)
                yield key
                # input
                # history
                # response

    return StreamingResponse(generate(), media_type="text/event-stream")


# 存在记忆 的输出
@app.post("/memory/ask")
def ask(body: dict):
    result = conversationChain.invoke(body['question'])
    # {'input': '请问你是谁啊', 'history': 'Human: 你是谁啊\nAI: 我是阿里云开发的一种人工智能模型，我叫通义千问。', 'response':
    # '我是阿里云开发的一种人工智能模型，我叫通义千问。'}
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
        # print(f"Hey, I received the name {input['name']}")
        print(input)
        return self._call_with_config(lambda x: x, input, config)


def get_history(docs):
    return {"chat_history": 'b64_images'}


# >>>>>>>>>>方法>>>>>>>>>>>>>>>
def call_llm(question: str):
    return cf_llm.invoke(question)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def stream_rag_prompt():
    return '''
    # Character
    You're a knowledgeable assistant capable of providing concise answers to a variety of questions, drawing from the context provided, and admitting when you don't know the answer.
    
    ## Skills
    1. **Answering Questions:** Utilize the given context to answer user questions. If the answer is not clear from the context, truthfully state that the answer is unknown to maintain accuracy in your responses.
    Question: {question}
    Context: {context}    
    
    ### Answering Questions Format:
    - Answer:  
    
    ## Constraints:
    - Keep answers to a maximum of three sentences to maintain brevity.
    - If the answer cannot be determined, simply confess that you do not know. Honesty is paramount in maintaining credibility.
    - If the answer is not reflected in the context, please reply: Sorry, I don't know for the moment.
    - Focus on gleaning answers from the context provided only.
    - All questions should be answered in Chinese
    '''


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


async def stream_wrapper(original_generator, citations):
    # 第一次yield返回这个
    yield f"data: {json.dumps({'citations': citations})}\n\n"
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
