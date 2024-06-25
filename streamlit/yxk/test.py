import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
)
from langchain_community.embeddings.cloudflare_workersai import (
    CloudflareWorkersAIEmbeddings,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from supabase.client import Client, create_client
from dotenv import load_dotenv
import os
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_community.llms.tongyi import Tongyi
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv(override=True)

account_id = os.getenv('CF_ACCOUNT_ID')
api_token = os.getenv('CF_API_TOKEN')
print(account_id)
print(api_token)

# CloudflareWorkersAI
model = '@cf/meta/llama-3-8b-instruct'
cf_llm = CloudflareWorkersAI(
    account_id=account_id,
    api_token=api_token,
    model=model
)

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
print(DASHSCOPE_API_KEY)

# qwen
qwen_llm = Tongyi(
    model='qwen2-1.5b-instruct'
)

api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')
print(api_key)
print(base_url)

# openai/moonshot
ms_llm = ChatOpenAI(
    openai_api_base=base_url,
    openai_api_key=api_key,
    model_name="moonshot-v1-8k",
    temperature=0.7,
)

# //维度是：1024，@cf/baai/bge-large-en-v1.5
# @cf/baai/bge-small-en-v1.5
embeddings = CloudflareWorkersAIEmbeddings(
    account_id=account_id,
    api_token=api_token,
    model_name="@cf/baai/bge-small-en-v1.5",
)

load_dotenv(override=True)

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_token = os.environ.get("SUPABASE_TOKEN")

print(f"supabase_url: " + supabase_url)
print(f"supabase_token: " + supabase_token)
print(f"supabase_key: " + supabase_key)

# https://supabase.com/dashboard/project/infrxrfaftyrxvkwvncf/editor/29610
supabase: Client = create_client(supabase_url, supabase_key)

# 将上述文件插入数据库。嵌入将自动为每个文档生成。
vectorstore_exist = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
retriever = vectorstore_exist.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_v2 = '''
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
'''


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": (retriever | format_docs), "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(prompt_v2)
        | qwen_llm
        | StrOutputParser()
)

# =======================================================
st.set_page_config(page_title="Welcome to ASL", layout="wide")

st.title("🤠 Welcome to ASL")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if cf_llm:
    with st.container():
        st.header("Chat with YXK")
        prompt = st.chat_input("Type something...")
        if prompt:
            st.markdown(prompt)
            ai_message = rag_chain.invoke(prompt + "请用中文回答")
            st.markdown(ai_message)
