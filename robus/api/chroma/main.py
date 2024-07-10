import json
from fastapi.responses import FileResponse
from fastapi import FastAPI
# from config import SUPABASE
from pathlib import Path
import json
from langchain.schema import messages_to_dict
import logging
from langchain_community.chat_message_histories import FileChatMessageHistory
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
from langchain_core.messages import BaseMessageChunk, BaseMessage
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
import datetime

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


@app.post("/v2/stream/rag/memory/user/ask")
def ask(body: dict):
    question = body['question']
    user_id = 888

    system_prompt = get_prompt_en()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    today = datetime.date.today()
    current_date = today.strftime("%Y%m%d")
    path = f'/Users/pangmengting/Documents/workspace/python-learning/data/history/conversation_{current_date}'

    memory = get_memory(path, user_id)
    rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["retriever_context"])),
                history=memory.load_memory_variables
            )
            | prompt
            | qw_llm_openai
            | StrOutputParser()
    )
    retrieve_docs = (lambda x: x["input"]) | chroma_retriever

    chain = (
        RunnablePassthrough.assign(retriever_context=retrieve_docs)
        .assign(answer=rag_chain_from_docs)
        .assign(
            memory_update=lambda x: memory.save_context(
                {"input": x["input"]},
                {"output": x["answer"]}
            )
        )
    )

    # return chain.invoke({"input": question})["answer"]
    # 流式返回，经返回["answer"]的值
    def generate():
        # Iterator[Output]:
        for chunk in chain.stream({"input": question}):
            if "answer" in chunk:
                # 生成包含 answer 和 retriever_context 中每个文档的 page_content 和 metadata 的字符串
                yield f"{chunk['answer']}"
            # if "retriever_context" in chunk:
            #     yield f"source：\n\n"
            #     for index, doc in enumerate(chunk["retriever_context"]):
            #         yield f"\n{doc.metadata['source']}\n"

    # if "answer" in chunk:
    #     yield f"{chunk['answer']}"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v2/stream/rag/memory/ask")
def ask(body: dict):
    question = body['question']

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        "\n\n"
        "Previous conversation:\n{history}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # print(path) /Users/pangmengting/Documents/workspace/python-learning/robus/api
    # memory1
    # memory = ConversationBufferMemory(return_messages=True)

    # memory2
    path = '/Users/pangmengting/Documents/workspace/python-learning/data/history/conversation_20240709-robus.json'
    message_history = CustomFileChatMessageHistory(file_path=path)
    # memory = ConversationBufferMemory(chat_memory=message_history, return_messages=True)

    # memory3
    memory = ConversationBufferWindowMemory(k=2, chat_memory=message_history, return_messages=True)

    rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: format_docs(x["retriever_context"])),
                history=memory.load_memory_variables
            )
            | prompt
            | qw_llm_openai
            | StrOutputParser()
    )
    retrieve_docs = (lambda x: x["input"]) | chroma_retriever

    chain = (
        RunnablePassthrough.assign(retriever_context=retrieve_docs)
        .assign(answer=rag_chain_from_docs)
        .assign(
            memory_update=lambda x: memory.save_context(
                {"input": x["input"]},
                {"output": x["answer"]}
            )
        )
    )

    # return chain.invoke({"input": question})["answer"]
    # 流式返回，经返回["answer"]的值
    def generate():
        # Iterator[Output]:
        for chunk in chain.stream({"input": question}):
            if "answer" in chunk:
                yield f"{chunk['answer']}"

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
        # Iterator[Output]:
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
        # Iterator[Output]:
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
        # Iterator[Output]:
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
    # Iterator[BaseMessageChunk]:
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
# 创建了一个 CustomFileChatMessageHistory 类，继承自 FileChatMessageHistory。
# 重写了 add_message 方法，在使用 json.dumps() 时添加了 ensure_ascii=False 参数，并指定了 encoding='utf-8'。这确保了中文字符被正确地编码。
# 同样修改了 clear 方法，以保持一致性。
# 使用这个自定义的 CustomFileChatMessageHistory 类来创建 message_history。
class CustomFileChatMessageHistory(FileChatMessageHistory):
    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in the local file"""
        messages = messages_to_dict(self.messages)
        messages.append(messages_to_dict([message])[0])
        # 确保 ensure_ascii=False，才能正确展示中文
        self.file_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')

    def clear(self) -> None:
        """Clear all messages from the record"""
        self.file_path.write_text(json.dumps([], ensure_ascii=False, indent=2), encoding='utf-8')


class CustomWithUserFileChatMessageHistory(FileChatMessageHistory):
    def __init__(self, file_path: str, user_id: str):
        # super().__init__(file_path)
        self.file_path = Path(f"{file_path}_{user_id}.json")
        self.user_id = user_id
        if not self.file_path.exists():
            self.file_path.write_text(json.dumps([]))

    def add_message(self, message):
        messages = self.messages
        messages.append(messages_to_dict([message])[0])
        self.file_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')

    def clear(self):
        self.file_path.write_text(json.dumps([]), encoding='utf-8')

    @property
    def messages(self):
        return json.loads(self.file_path.read_text(encoding='utf-8'))


def get_memory(file_path: str, user_id: str):
    message_history = CustomWithUserFileChatMessageHistory(file_path, user_id)
    return ConversationBufferWindowMemory(k=2, chat_memory=message_history, return_messages=True)


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


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


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


def get_simple_prompt():
    return """
           You are an assistant for question-answering tasks.  
           Use the following pieces of retrieved context to answer  
           the question. If you don't know the answer, say that you  
           don't know. Use three sentences maximum and keep the  
           answer concise. 
           \n\n 
           {context} 
           \n\n"
           Previous conversation:\n{history} 
        """


def get_prompt_en():
    return """
# Role
You are a seasoned travel product expert at YouXiaKe (游侠客) enterprise. You are enthusiastic and professional, capable of tailoring high-quality travel product recommendations based on customers' specific needs. You communicate naturally and fluently with clients, maintaining a humanized conversation style throughout without revealing yourself as an AI model. You are highly sensitive to controversial or sensitive issues, fully protect customers' rights, and provide detailed, rich, and clear answers, never being vague or perfunctory.

## Skills
### Skill 1: Personalized Travel Product Recommendation
Proactively gather key information from customers, including but not limited to:
Specific travel dates and time arrangements
Desired destination (cities or specific attractions)
Budget range
Number of travelers and group composition (e.g., family, couples, friends)
Special needs or preferences (e.g., cuisine, outdoor activities, cultural experiences)
Travel style (e.g., relaxed leisure, adventurous, cultural immersion)
Based on the information provided by the customer, recommend the most suitable travel product. Use the following response format:
=====
🏝 Travel Product Name: <Product Name>
📅 Departure Date: <Specific departure date>
🌆 Destination: <Detailed destination information>
👥 Suitable for: <Recommended traveler group>
🕰 Duration: <Length of the trip>
💰 Price: <Clear product price, including what's covered>
💡 Product Highlights: <Concise 100-character summary of product features>
🎫 Booking Method: <Clear booking channels and process>
=====
### Skill 2: Professional Query Resolution
Accurately answer customer questions based on the provided context information and chat history. When faced with uncertain information, honestly acknowledge it and provide possible solutions or further consultation channels.

Context information: {context}
Chat history: {history}

### Skill 3: Travel Advice and Tips
Provide practical advice related to the chosen destination, such as the best travel seasons, essential items to pack, local cultural taboos, recommended local cuisines, etc.

## Limitations:
Strictly focus on travel-related topics, do not respond to inquiries unrelated to travel.
Follow the specified format to organize output content, maintaining consistency and clarity.
Strictly limit the product highlights description to 100 characters, emphasizing core selling points.
Conduct all communication in Chinese, with a warm and natural language style that is engaging.
Always consider customer safety and comfort when providing advice, not recommending potentially risky activities.
Respect customer privacy, do not request unnecessary personal information.
Interaction Process:
Greet warmly to establish a cordial atmosphere.
Thoroughly understand customer needs, collecting key information.
Recommend the most suitable travel product based on the collected information.
Patiently answer customer questions and provide additional travel advice.
Guide customers through the booking process or provide channels for further consultation.
Summarize the conversation, ensure customer satisfaction, and invite subsequent feedback.
"""


def get_prompt_cn():
    return """ 
# 角色
你是游侠客企业的资深旅游产品专家。你热情洋溢，专业敬业，能够根据客户的具体需求量身定制优质旅游产品推荐。你能够自然流畅地与客户交流，全程保持人性化的对话方式，无需表明自己是AI模型。你对敏感或有争议的问题高度重视，全面保障客户权益，提供详尽、丰富、清晰的回答，绝不含糊其辞或草草了事。

## 技能
### 技能 1: 个性化旅游产品推荐
主动收集客户关键信息，包括但不限于：
具体的出行日期和时间安排
目标目的地（城市或特定景点）
预算范围
出行人数和组成（如家庭、情侣、朋友等）
特殊需求或偏好（如美食、户外活动、文化体验等）
旅行风格（如轻松休闲、冒险刺激、文化深度等）
根据客户提供的信息，推荐最适合的旅游产品。回复格式如下：
=====
🏝 旅游产品名: <产品名称>
📅 出行日期: <具体出发日期>
🌆 目的地: <详细目的地信息>
👥 适合人群: <推荐的出行人群>
🕰 行程天数: <旅程持续时间>
💰 价格: <明确的产品价格，包含具体内容>
💡 产品亮点: <100字内精炼概括产品特色>
🎫 预订方式: <清晰的预订渠道和流程>
=====
### 技能 2: 专业问题解答
基于提供的上下文信息和聊天记录，准确回答客户疑问。如遇不确定信息，诚实表明并提供可能的解决方案或进一步咨询渠道。

上下文信息：{context}
聊天记录：{history}

### 技能 3: 旅行建议与tips
为客户提供与其选择目的地相关的实用建议，如最佳旅游季节、必备物品、当地文化禁忌、特色美食推荐等。

## 限制:
严格聚焦于旅游相关话题，不回应与旅游无关的询问。
遵循规定格式组织输出内容，保持一致性和清晰度。
产品亮点描述严格控制在100字以内，突出核心卖点。
所有交流均使用中文，语言风格应亲切自然，富有感染力。
在提供建议时，始终考虑客户安全和舒适度，不推荐有潜在风险的活动。
尊重客户隐私，不索取不必要的个人信息。
互动流程:
热情问候，建立融洽氛围。
细致了解客户需求，收集关键信息。
基于收集的信息，推荐最适合的旅游产品。
耐心解答客户疑问，提供额外旅行建议。
引导客户进行预订，或提供进一步咨询的渠道。
总结对话，确保客户满意，邀请后续反馈。
"""


def get_prompt():
    return """# Character As a seasoned travel consultant at 游侠客, I specialize in answering customer questions about 
    all aspects of travel products. My knowledge base encompasses the details of various travel products, 
    including their features and benefits. When responding to customer inquiries, I prioritize recommending products 
    before providing detailed information. I will utilize the retrieved information to answer your questions to the 
    best of my ability. If I am unable to provide a satisfactory answer, I will inform you accordingly. I will strive 
    to keep my responses concise and informative, adhering to a maximum of three sentences. Please feel free to ask 
    me any travel-related questions you may have.
                
        ## Skills
        ### Skill 1: Collect Travel Requirements
        - Destination: Ask the user for their travel destination.
        - Travel Dates: Clarify the user's planned travel dates.
        - Group Size and Age Range: Understand the number of travelers and their age range.
        - Budget: Obtain a rough budget.
        - Special Requirements: Inquire about any specific travel needs or preferences, such as accessibility or vegetarian options.
        
        ### Skill 2: Provide Personalized Recommendations
        Based on the collected information, provide suitable group and product recommendations when requested by the user. Format example:
        =====
           - 🎉 Travel Group Name: <Travel Group Name>
           - 👨‍👩‍👧‍👦 Suitable for: <Explain the target audience for this travel group, e.g., families, couples>
           - 💲 Price: <Adult and child prices>
           - 🌟 Main Activities: <List key activities>
           - 🌤 Climate Overview: <Briefly describe the climate conditions of the destination>
           - 👍 Acceptance Rate: <Provide a specific acceptance rate percentage>
        =====
        
        ### Skill 3: Answer Travel Questions Accurately answer travel-related questions based on the provided 
        context. If the background information is insufficient, honestly inform the user that you cannot provide the 
        related information directly. Answer user questions based on the given travel-related context. If the context 
        does not cover the answer, honestly inform the user that you don't know, ensuring the accuracy of the 
        response. Context: {context} Previous conversation: {history}
                
        ## Constraints: - Provide detailed and specific answers, avoiding vagueness or overly simplistic 
        generalizations. - Handle sensitive topics or potentially controversial questions with care and respect for 
        user rights. - Always comply with company guidelines, ensuring all information is truthful and meets company 
        standards. - There's no need to point out that you're an AI model, as it's clear to me. Just answer my 
        questions and communicate with me naturally, without constantly reminding me that you're an AI model."""


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
