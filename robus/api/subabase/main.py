from fastapi import FastAPI

import logging

log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "subabase"}



