from chainlit.types import (
    AskFileResponse
)
import chainlit as cl
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embedding = CloudflareWorkersAIEmbeddings(
    account_id=os.getenv('CF_ACCOUNT_ID'),
    api_token=os.getenv('CF_API_TOKEN'),
    model_name="@cf/baai/bge-small-en-v1.5",
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def store_uploaded_file(uploaded_file: AskFileResponse):
    file_path = f"./tmp/{uploaded_file.name}"
    open(file_path, "wb").write(uploaded_file.content)
    return file_path


@cl.on_chat_start
async def init():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!", accept=["application/pdf"]
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    file_path = store_uploaded_file(file)

    # Load PDF file into documents
    docs = PyMuPDFLoader(file_path).load()

    msg = cl.Message(content=f'You have {len(docs)} document(s) in the PDF file.')
    await msg.send()

    # Split the documents into chunks
    split_docs = text_splitter.split_documents(docs)

    # Create a Chroma vector store
    embeddings = embedding
    docsearch = await cl.make_async(Chroma.from_documents)(
        split_docs, embeddings, collection_name=file.name
    )

    qw_llm_openai = ChatOpenAI(
        openai_api_base=os.getenv('DASHSCOPE_API_BASE'),
        openai_api_key=os.getenv('DASHSCOPE_API_KEY'),
        model_name="qwen2-1.5b-instruct",
        temperature=0,
        streaming=True,
    )
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        qw_llm_openai,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Let the user know that the system is ready
    await msg.update(content=f"`{file.name}` processed. You can now ask questions!")

    return chain


@cl.on_chat_end
async def process_response(res):
    answer = res["answer"]
    await cl.Message(content=answer).send()
