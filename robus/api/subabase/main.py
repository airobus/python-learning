import json
from fastapi.responses import FileResponse
from fastapi import FastAPI
# from config import SUPABASE
from pathlib import Path
import logging
from robus.config import SUPABASE

log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "subabase"}

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
    return {'aa', 'sss'}
