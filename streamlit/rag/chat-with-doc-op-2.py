from operator import itemgetter
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, format_document, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

qw_llm_openai = ChatOpenAI(
    openai_api_base=os.getenv('DASHSCOPE_API_BASE'),
    openai_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_name="qwen2-1.5b-instruct",
    temperature=0,
    streaming=True,
)
qw_embedding = DashScopeEmbeddings(
    model="text-embedding-v2", dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
)

DATA_DIR = '/Users/pangmengting/Documents/workspace/python-learning/data'
CHROMA_DATA_PATH = f"{DATA_DIR}/chroma_vector_db"
CHROMA_CLIENT = chromadb.PersistentClient(
    path=CHROMA_DATA_PATH,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)

fix_collection_name = 'yxk-know-index-3'
vectordb = Chroma(collection_name=fix_collection_name,
                  client=CHROMA_CLIENT,
                  embedding_function=qw_embedding)

retriever = vectordb.as_retriever()

# Define prompts
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based only on the following context:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("user", "{question}"),
    ]
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | qw_llm_openai
        | StrOutputParser(),
    ),
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | qw_llm_openai | StrOutputParser()

# Streamlit app
st.title("Chat History and Question Processor")

chat_history = []
if 'chat_history' in st.session_state:
    chat_history = st.session_state['chat_history']

# Sidebar for file upload and input
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx"])


st.sidebar.header("Chat Input")
st.sidebar.subheader("Chat History")
user_input = st.sidebar.text_area("Enter your chat history (one message per line)")

if user_input:
    chat_history = [tuple(line.split(':', 1)) for line in user_input.split('\n') if ':' in line]
    st.session_state['chat_history'] = chat_history

question = st.sidebar.text_input("Enter your question")

if st.sidebar.button("Submit"):
    with st.spinner("Processing..."):
        chat_history_data = {
            "chat_history": chat_history,
            "question": question
        }
        result = chain.invoke(chat_history_data)
        st.write("Answer:", result)
