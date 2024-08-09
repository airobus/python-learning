import os
import tempfile
from operator import itemgetter
from typing import List, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    PyMuPDFLoader,
    CSVLoader,
    BSHTMLLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    YoutubeLoader,
    OutlookMessageLoader,
    JSONLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, format_document
from langchain_core.runnables import RunnableConfig, RunnableParallel, RunnableBranch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files, openai_api_key):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        # loader = PyPDFLoader(temp_filepath)
        unsanitized_filename = file.name
        filename = os.path.basename(unsanitized_filename)
        loader, known_type = get_loader(filename, file.type, temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    qw_embedding = DashScopeEmbeddings(
        model="text-embedding-v2", dashscope_api_key=openai_api_key
    )

    fix_collection_name = 'yxk-know-index-3'
    persist_directory = '/Users/pangmengting/Documents/workspace/python-learning/data/chroma_vector_db'
    vectorstore = Chroma.from_documents(
        splits,
        qw_embedding,
        collection_name=fix_collection_name,
        persist_directory=persist_directory
    )

    texts = [doc.page_content for doc in splits]

    bm25_retriever = BM25Retriever.from_texts(
        texts=texts,
    )

    # chroma_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "fetch_k": 4})
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    )
    print('======>configure_retriever ok ok ok ')
    return ensemble_retriever


def get_loader(filename: str, file_content_type: str, file_path: str):
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    known_source_ext = [
        "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css", "cpp", "hpp", "h", "c", "cs", "sql", "log",
        "ini", "pl", "pm", "r", "dart", "dockerfile", "env", "php", "hs", "hsc", "lua", "nginxconf", "conf", "m", "mm",
        "plsql", "perl", "rb", "rs", "db2", "scala", "bash", "swift", "vue", "svelte", "msg",
    ]

    if file_ext == "pdf":
        # loader = PyMuPDFLoader(file_path)
        loader = PyPDFLoader(file_path)
    elif file_ext == "csv":
        loader = CSVLoader(file_path)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(file_path, mode="elements")
    elif file_ext == "xml":
        loader = UnstructuredXMLLoader(file_path)
    elif file_ext in ["htm", "html"]:
        loader = BSHTMLLoader(file_path, open_encoding="unicode_escape")
    elif file_ext == "md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(file_path)
    elif (
            file_content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or file_ext in ["doc", "docx"]
    ):
        loader = Docx2txtLoader(file_path)
    elif file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ] or file_ext in ["xls", "xlsx"]:
        loader = UnstructuredExcelLoader(file_path)
    elif file_content_type in [
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ] or file_ext in ["ppt", "pptx"]:
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_ext == "msg":
        loader = OutlookMessageLoader(file_path)
    elif file_ext in known_source_ext or (
            file_content_type and file_content_type.find("text/") >= 0
    ):
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        loader = TextLoader(file_path, autodetect_encoding=True)
        known_type = False

    return loader, known_type


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**正在检索信息**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**问题:** {query}")
        self.status.update(label=f"**检索信息:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx + 1} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def get_prompt_cn():
    return """ 
# 角色
你是游侠客企业经验丰富、热情洋溢且专业的旅游产品专家，熟知游侠客所有旅游产品的详细资讯。能以亲切、自然且人性化的方式与客户无障碍交流，始终把保障客户权益放在首位，给出清晰、丰富且毫无歧义的回答。

## 技能
### 技能 1: 专业问题解答
依据给定的上下文信息和聊天记录，精确且详尽地回应客户关于旅游产品的疑问。着重留意日期、地点、价格、产品名称等关键要素。若信息不清晰，需诚恳告知，并提供可能的解决办法或进一步咨询的途径。回复示例：
=====
    - 对于您询问的[具体问题]，目前的状况是[具体回答]。倘若信息不准确，您能够通过[解决办法或咨询途径]获取更确切的信息。
=====

### 技能 2: 个性化旅游产品推荐
积极主动地收集客户的关键信息，涵盖但不限于：
- 明确的出行日期及详细的时间安排
- 确切的目标目的地（精确至城市或特定景点）
- 清晰的预算范围
- 出行人数及构成（如家庭、情侣、朋友等）
- 特殊需求或偏好（比如美食、户外活动、文化体验等）
- 特定的旅行风格（像轻松休闲、冒险刺激、文化深度等）
根据上述获取到的信息及上下文，提取关键要点，如产品名称、详细目的地信息、行程天数、价格等。为客户推荐最为适配的旅游产品。回复格式如下：
=====
🏝 旅游产品名: <产品名称>
🌆 目的地: <详细目的地信息>
🕰 行程天数: <旅程持续时间，几天几夜>
💰 价格: <产品价格，包含成人价格和儿童价格>
=====

### 技能 3: 旅行建议与 tips
为客户提供其所选目的地的实用指南，包含最佳旅游季节、必备物品、当地文化禁忌、特色美食推荐等方面。回复示例：
=====
    - 有关[目的地]，最佳旅游季节是[具体季节]，您需要准备[必备物品]，同时要留意当地的文化禁忌[列举禁忌]，特色美食有[美食推荐]。
=====

### 技能 4: 使用以下在 <context></context> XML 标签内的上下文作为您学到的知识。
<context>
    {context}\n
    {chat_history}
</context>
鉴于上下文信息，回答查询。
 
## 限制:
若依据现有信息无法回答客户问题，直接回复“不知道”。
只回应与旅游相关的客户咨询，避免无关及过度的回答，不重复作答。
严格围绕旅游主题进行交流，不涉及无关话题。
按照规定格式组织输出内容，保证一致性与清晰度。
产品亮点描述控制在 50 字以内，突出核心卖点。
全程使用中文与客户交流，语言亲切自然且富有感染力。
提供建议时，着重考虑客户的安全与舒适度，不推荐存在潜在风险的活动。
尊重客户隐私，不索要非必要的个人信息。
互动流程:
热情友好地问候客户，营造轻松愉悦的交流氛围。
细致全面地了解客户需求，精准收集关键信息。
依据收集到的信息，为客户推荐适宜的旅游产品。
耐心解答客户的疑问，给予实用的旅行建议。
引导客户进行预订，或提供进一步咨询的渠道。
总结对话内容，确保客户满意，邀请客户进行后续反馈。
"""


def format_docs(docs):
    context = "\n".join(doc.page_content for doc in docs)
    return context


openai_api_key = st.sidebar.text_input("Qwen API Key", type="password", placeholder="随便输入即可使用")
# openai_api_base = st.sidebar.text_input("Qwen API Base", type="default")
if not openai_api_key:
    st.info("请随便输入API key得以继续.")
    # st.stop()

load_dotenv(override=True)
openai_api_key = os.getenv('DASHSCOPE_API_KEY')
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF/Markdown/Text files", type=["pdf", "md", "text", "txt"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("请上传文档继续.")
    st.stop()

retriever = configure_retriever(uploaded_files, openai_api_key)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

llm = ChatOpenAI(
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    openai_api_key=openai_api_key,
    model_name="qwen2-1.5b-instruct",
    temperature=0,
    streaming=True,
)

_template = """Given the following conversation and a follow up question.
Please reply strictly in Chinese.

Chat History:
{chat_history}
Follow Up Input: {question}
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Answer the question based only on the following context:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm, retriever=retriever, memory=memory, verbose=True, condense_question_prompt=CONDENSE_QUESTION_PROMPT
# )

if len(msgs.messages) == 0 or st.sidebar.button("清空聊天历史"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
    user_query = "请用中文回答：" + user_query

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = chain.invoke({"question": user_query, "chat_history": []},
                                config=RunnableConfig(callbacks=[retrieval_handler, stream_handler]))
