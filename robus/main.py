#!/usr/bin/env python3
import hashlib
import uuid
from datetime import datetime

from chromadb.utils.batch_utils import create_batches
from langchain_text_splitters import RecursiveCharacterTextSplitter

from public.usage import USAGE as html
import logging
from api.hello import router as hello_router
from api.subabase.main import app as subabase_router
from api.chroma.main import app as chroma_router
import asyncio
import uvicorn
import os
from typing import AsyncIterable, Awaitable, Optional, Tuple
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from config import AppConfig, PersistentConfigTest, STATIC_DIR, UPLOAD_DIR, CHROMA_CLIENT, embeddings, CHUNK_SIZE, \
    CHUNK_OVERLAP
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
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
)

# >>>>>>>>>>基础>>>>>>>>>>>>>>
log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory=STATIC_DIR), name="public")

# 两种方式
app.include_router(hello_router, prefix="/hello")
app.mount("/test", hello_router)
# 核心
app.mount("/subabase", subabase_router)
app.mount("/chroma", chroma_router)

load_dotenv(override=True)

# 在应用程序启动时执行的初始化操作，方式1
app.state.config = AppConfig()
app.state.config.PersistentConfigTest = PersistentConfigTest
app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP


# 在异步编程中，await关键字的作用就是等待其后的异步操作（如另一个async函数的调用或一个异步IO操作）完成，之后才会继续执行紧跟在await之后的代码。
# 这意味着在await表达式处，当前的异步函数会暂停执行，控制权返回到事件循环，允许其他任务（如果有）在这段时间内运行。
# 一旦等待的操作完成，该异步函数会恢复执行，从await之后的下一条语句继续。
async def wait_done(fn: Awaitable, event: asyncio.Event):
    try:
        await fn
    except Exception as e:
        print(e)
        # 设置事件通知等待的其他任务
        event.set()
    finally:
        # 设置事件通知等待的其他任务
        event.set()


# 流式输出的一种方式
async def call_llm(question: str, prompt: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()

    qw_llm_openai = ChatOpenAI(
        openai_api_base=os.getenv('DASHSCOPE_API_BASE'),
        openai_api_key=os.getenv('DASHSCOPE_API_KEY'),
        model_name="qwen2-1.5b-instruct",
        temperature=0.7,
        streaming=True,
        callbacks=[callback]
    )
    coroutine = wait_done(qw_llm_openai.agenerate(messages=[[HumanMessage(content=question)]]), callback.done)

    # qw_llm = Tongyi(
    #     model='qwen2-1.5b-instruct',
    #     streaming=True,
    #     callbacks=[callback]
    # )
    # prompts = [question]
    #
    # coroutine = wait_done(qw_llm.agenerate(prompts=prompts), callback.done)

    task = asyncio.create_task(coroutine)

    async for token in callback.aiter():
        yield f"{token}"

    await task


# @app.get("/")
# def _root():
#     return Response(content=html, media_type="text/html")


@app.post("/ask")
def ask(body: dict):
    return StreamingResponse(call_llm(body['question'], body['prompt']), media_type="text/event-stream")


@app.get("/")
async def homepage():
    return FileResponse('public/index.html')


# 显示的指明favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))


@app.get("/upload/page")
async def homepage():
    return FileResponse('public/upload.html')


@app.post("/upload/file")
async def upload_file(
        collection_name: Optional[str] = Form(None),
        file: UploadFile = File(...)
):
    log.info(f"file.content_type: {file.content_type}")
    try:
        unsanitized_filename = file.filename
        filename = os.path.basename(unsanitized_filename)

        file_path = f"{UPLOAD_DIR}/{filename}"
        print(file_path)
        contents = file.file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            f.close()

        f = open(file_path, "rb")
        if collection_name is None:
            collection_name = calculate_sha256(f)[:63]
        f.close()

        loader, known_type = get_loader(filename, file.content_type, file_path)
        data = loader.load()

        try:
            result = store_data_in_vector_db(data, collection_name)

            if result:
                return {
                    "status": True,
                    "collection_name": collection_name,
                    "filename": filename,
                    "known_type": known_type,
                }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=e,
            )
    except Exception as e:
        log.exception(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pandoc is not installed on the server. Please contact your administrator for assistance.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=lambda err="": f"Something went wrong :/\n{err if err else ''}",
            )


def store_data_in_vector_db(data, collection_name, overwrite: bool = False) -> tuple[bool, None]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app.state.config.CHUNK_SIZE,
        chunk_overlap=app.state.config.CHUNK_OVERLAP,
        add_start_index=True,
    )

    docs = text_splitter.split_documents(data)

    if len(docs) > 0:
        log.info(f"store_data_in_vector_db {docs}")
        return store_docs_in_vector_db(docs, collection_name, overwrite, embeddings), None
    else:
        raise ValueError(
            "The content provided is empty. Please ensure that there is text or data present before proceeding.")


def store_docs_in_vector_db(docs, collection_name, overwrite: bool = False, embeddings=None) -> bool:
    log.info(f"store_docs_in_vector_db {docs} {collection_name}")

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    log.info(f"metadatas==> {metadatas}")
    # ChromaDB does not like datetime formats
    # for meta-data so convert them to string.
    for metadata in metadatas:
        for key, value in metadata.items():
            if isinstance(value, datetime):
                metadata[key] = str(value)

    try:
        if overwrite:
            for collection in CHROMA_CLIENT.list_collections():
                if collection_name == collection.name:
                    log.info(f"deleting existing collection {collection_name}")
                    CHROMA_CLIENT.delete_collection(name=collection_name)

        collection = CHROMA_CLIENT.create_collection(name=collection_name)
        embedding_texts = list(map(lambda x: x.replace("\n", " "), texts))
        embedd = embeddings.embed_documents(embedding_texts)

        for batch in create_batches(
                api=CHROMA_CLIENT,
                ids=[str(uuid.uuid4()) for _ in texts],
                metadatas=metadatas,
                embeddings=embedd,
                documents=texts,
        ):
            # 最终向量化存储
            collection.add(*batch)

        return True
    except Exception as e:
        log.exception(e)
        if e.__class__.__name__ == "UniqueConstraintError":
            return True

        return False


def get_embedding_function(
        embedding_function,
):
    return lambda query: embedding_function.encode(query).tolist()


def get_loader(filename: str, file_content_type: str, file_path: str):
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    known_source_ext = [
        "go",
        "py",
        "java",
        "sh",
        "bat",
        "ps1",
        "cmd",
        "js",
        "ts",
        "css",
        "cpp",
        "hpp",
        "h",
        "c",
        "cs",
        "sql",
        "log",
        "ini",
        "pl",
        "pm",
        "r",
        "dart",
        "dockerfile",
        "env",
        "php",
        "hs",
        "hsc",
        "lua",
        "nginxconf",
        "conf",
        "m",
        "mm",
        "plsql",
        "perl",
        "rb",
        "rs",
        "db2",
        "scala",
        "bash",
        "swift",
        "vue",
        "svelte",
        "msg",
    ]

    if file_ext == "pdf":
        loader = PyPDFLoader(
            file_path, extract_images=app.state.config.PDF_EXTRACT_IMAGES
        )
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


# print(app.state.config.PersistentConfigTest)

def calculate_sha256(file):
    sha256 = hashlib.sha256()
    # Read the file in chunks to efficiently handle large files
    for chunk in iter(lambda: file.read(8192), b""):
        sha256.update(chunk)
    return sha256.hexdigest()


def calculate_sha256_string(string):
    # Create a new SHA-256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the bytes of the input string
    sha256_hash.update(string.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    hashed_string = sha256_hash.hexdigest()
    return hashed_string


# 在应用程序启动时执行的初始化操作，方式2
@app.on_event("startup")
async def startup_event():
    # 例如，连接到数据库、加载配置、初始化缓存等
    log.info(" log-Initializing data...")


if __name__ == "__main__":
    uvicorn.run(host="127.0.0.1", port=9999, app=app)
