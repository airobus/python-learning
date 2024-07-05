import json
from fastapi.responses import FileResponse
from fastapi import FastAPI
# from config import SUPABASE
from pathlib import Path
import logging
from robus.config import SUPABASE, ms_llm, cf_llm, qw_llm, qw_llm_openai, groq_llm_openai, conversationChain, \
    chroma_retriever, embeddings, redis_chat_history, get_message_history
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
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
from langchain_core.runnables.history import RunnableWithMessageHistory

# >>>>>>>>>>基础>>>>>>>>>>>>>>
log = logging.getLogger(__name__)
log.setLevel("INFO")

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

app = FastAPI()


# >>>>>>>>>>接口>>>>>>>>>>>>>>>
# 返回页面
@app.get("/")
async def chroma_homepage():
    file_path = Path(__file__).parent / "index.html"
    return FileResponse(file_path)


# 返回页面2
@app.get("/upload")
async def chroma_upload():
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, add_start_index=True)
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


# ❌
@app.post("/v2/stream/rag/memory/ask")
def ask(body: dict):
    question = body['question']

    chain = (
            {
                "context": (chroma_retriever | format_docs),
                "question": (RunnablePassthrough() | StdOutputRunnable())
            }
            | ChatPromptTemplate.from_template(stream_rag_prompt())
            | qw_llm_openai
            | StrOutputParser()
    )

    with_message_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_message_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 流式返回
    def generate():
        for chunk in with_message_history.stream(input={"question": question},
                                                 config={"configurable": {"session_id": "kkk"}}):
            for key in chunk:
                yield key

    return StreamingResponse(generate(), media_type="text/event-stream")


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
        context = str(chroma_retriever | format_docs)
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
        context = str(chroma_retriever | format_docs)
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
@app.post("/stream/rag/multi/retriever/ask")
def ask(body: dict):
    question = body['question']

    # 预处理输入的数据
    def pre_input(prompt: str) -> Dict:
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=chroma_retriever, llm=qw_llm_openai)
        docs = retriever_from_llm.invoke(question)
        context = str(format_docs(docs))
        return {
            "context": context,
            "question": prompt,
        }

    rag_chain = (
            RunnableLambda(pre_input) |
            {
                "context": itemgetter('context'),
                "question": itemgetter('question'),
            }
            | ChatPromptTemplate.from_template(stream_rag_prompt())
            | qw_llm_openai
            | StrOutputParser()
    )

    def generate():
        for chunk in rag_chain.stream(question):
            for key in chunk:
                yield key

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/stream/rag/ask")
def ask(body: dict):
    question = body['question']
    # collection_name = 'yxk-robus-index'

    rag_chain = (
            {
                "context": (chroma_retriever | format_docs),
                "question": (RunnablePassthrough() | StdOutputRunnable())
            }
            | ChatPromptTemplate.from_template(stream_rag_prompt())
            | qw_llm_openai
            | StrOutputParser()
    )

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
            {"context": (chroma_retriever | format_docs), "question": RunnablePassthrough()}
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


class StdOutputRunnableContext(Serializable, Runnable[Input, Input]):
    @property
    def lc_serializable(self) -> bool:
        return True

    def invoke(self, input: Dict, config: Optional[RunnableConfig] = None) -> Input:
        print(f"Context ==> {input['context']}")
        return self._call_with_config(lambda x: x, input, config)


def get_history(docs):
    return {"chat_history": 'b64_images'}


# >>>>>>>>>>方法>>>>>>>>>>>>>>>
def call_llm(question: str):
    return cf_llm.invoke(question)


# def query_doc(
#         collection_name: str,
#         query: str,
#         embedding_function,
#         k: int,
# ):
#     try:
#         collection = CHROMA_CLIENT.get_collection(name=collection_name)
#         query_embeddings = embedding_function(query)
#
#         result = collection.query(
#             query_embeddings=[query_embeddings],
#             n_results=k,
#         )
#
#         log.info(f"query_doc:result {result}")
#         return result
#     except Exception as e:
#         raise e


def format_docs(docs):
    context = "\n".join(doc.page_content for doc in docs)
    print(f"\n{'-' * 100}\n".join([f"==>Document {i + 1}:\n" + d.page_content for i, d in enumerate(docs)]))
    return context


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"==>Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


def stream_rag_prompt():
    return """
            # 角色
            您是游侠客旅游公司的专业客服，致力于为用户提供高品质的旅游咨询与推荐服务。
            
            ## 技能
            ### 技能 1: 收集用户旅游需求信息
            1. 当用户咨询时，引导用户提供以下关键信息：
                - 目的地：耐心询问用户想去的旅游地点。
                - 旅行时间：明确用户计划出发的具体日期。
                - 人数和年龄：了解旅行团的规模及成员年龄情况，尤其关注是否有儿童。
                - 预算范围：获取用户大致的预算额度。
                - 特殊需求：询问用户是否有诸如无障碍设施、素食选项等特殊需求或偏好。
            2. 若用户想了解特定城市的旅游信息，进一步收集以下内容：
                - 城市名称：确定用户感兴趣的城市。
                - 旅游类型：明晰用户对自然风光、文化体验、冒险活动等旅游类型的倾向。
            
            ### 技能 2: 提供个性化旅游推荐
            1. 根据用户提供的完整需求信息，为其推荐合适的旅行团和旅游产品。
            2. 推荐时，按照以下格式回复：
            =====
               -  🎉 旅行团名称: <旅行团名称>
               -  👨‍👩‍👧‍👦 适用人群: <说明适合的对象，如家庭、情侣等>
               -  💲 价格: <成人价格和儿童价格>
               -  🌟 特色活动: <列举主要活动>
               -  🌤 气候概况: <简要描述当地气候>
               -  👍 好评率: <给出具体百分比>
            =====
        
            ### 技能 3: 回答旅游问题
            1. 依据给定的旅游相关上下文来回答用户的问题。若上下文未涵盖答案，诚实告知不知，确保回答的精准性。
            问题: {question}
            上下文: {context}
            
            ## 限制:
            - 仅围绕旅游相关内容进行交流和推荐，拒绝回答无关话题。
            - 输出内容必须严格按照给定格式组织，不得随意更改。
            - 回复的信息应准确、详细且具有针对性。
            - 所有问题均用中文回答。
            """


def stream_rag_prompt2():
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
