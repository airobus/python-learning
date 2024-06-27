import json
from fastapi.responses import FileResponse
from fastapi import FastAPI
# from config import SUPABASE
from pathlib import Path
import logging
from robus.config import SUPABASE, ms_llm, cf_llm, qw_llm
from fastapi.responses import Response, StreamingResponse

log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()


@app.get("/")
async def subabase_homepage():
    file_path = Path(__file__).parent / "index.html"
    return FileResponse(file_path)


@app.get("/select")
def select():
    response = SUPABASE.table('documents').select("*").execute()
    return response


@app.post("/ask")
def ask(body: dict):
    return Response(call_llm(body['question']))


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


def call_llm(question: str):
    return cf_llm.invoke(question)


async def openai_stream_wrapper(original_generator, citations):
    yield f"data: {json.dumps({'citations': citations})}\n\n"
    async for data in original_generator:
        yield data


async def ollama_stream_wrapper(original_generator, citations):
    yield f"{json.dumps({'citations': citations})}\n"
    async for data in original_generator:
        yield data
