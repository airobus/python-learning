import json

from fastapi import FastAPI
from config import SUPABASE

import logging

log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "subabase"}


@app.get("/select")
def select():
    response = SUPABASE.table('documents').select("*").execute()
    return response
