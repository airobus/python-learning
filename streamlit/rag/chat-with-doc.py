import os
import tempfile
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
# from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
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
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

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
    print("========》》》Temporary directory path:", temp_dir.name)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    qw_embedding = DashScopeEmbeddings(
        model="text-embedding-v2", dashscope_api_key=openai_api_key
    )
    # vectordb = DocArrayInMemorySearch.from_documents(splits, qw_embedding)

    fix_collection_name = 'yxk-know-index-3'
    persist_directory = '/Users/pangmengting/Documents/workspace/python-learning/data/chroma_vector_db'
    vectorstore = Chroma.from_documents(
        splits,
        qw_embedding,
        collection_name=fix_collection_name,
        persist_directory=persist_directory
    )

    # Define retriever
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


def get_loader(filename: str, file_content_type: str, file_path: str):
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    known_source_ext = [
        "go", "py", "java", "sh", "bat", "ps1", "cmd", "js", "ts", "css", "cpp", "hpp", "h", "c", "cs", "sql", "log",
        "ini", "pl", "pm", "r", "dart", "dockerfile", "env", "php", "hs", "hsc", "lua", "nginxconf", "conf", "m", "mm",
        "plsql", "perl", "rb", "rs", "db2", "scala", "bash", "swift", "vue", "svelte", "msg",
    ]

    if file_ext == "pdf":
        loader = PyMuPDFLoader(file_path)
        # loader = PyPDFLoader(
        #     file_path, extract_images=app.state.config.PDF_EXTRACT_IMAGES
        # )
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


openai_api_key = st.sidebar.text_input("Qwen API Key", type="password", placeholder="随便输入即可使用")
# openai_api_base = st.sidebar.text_input("Qwen API Base", type="default")
if not openai_api_key:
    st.info("请随便输入API key得以继续.")
    # st.stop()

openai_api_key = 'sk-c9b505536e9d4a86a939b8ce7f10ff4f'
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

# Setup LLM and QA chain
# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
# )

llm = ChatOpenAI(
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    openai_api_key=openai_api_key,
    model_name="qwen2-1.5b-instruct",
    temperature=0,
    streaming=True,
)

# llm = ChatOpenAI(
#     openai_api_base=openai_api_base,
#     openai_api_key=openai_api_key,
#     model_name="moonshot-v1-8k",
#     temperature=0,
#     streaming=True,
# )

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in chinese language.
Please reply strictly in Chinese.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, condense_question_prompt=CONDENSE_QUESTION_PROMPT
)

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
        # response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        response = qa_chain.invoke(user_query, config=RunnableConfig(callbacks=[retrieval_handler, stream_handler]))
