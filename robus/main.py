#!/usr/bin/env python3
from public.usage import USAGE as html

from fastapi import FastAPI
from fastapi.responses import Response
import logging
from api.hello import router as hello_router
from api.subabase.main import app as subabase_router
import os
from dotenv import load_dotenv
from supabase.client import Client, create_client

from robus.config import AppConfig

log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()

# 两种方式
app.include_router(hello_router, prefix="/hello")
app.mount("/test", hello_router)
# 核心
app.mount("/subabase", subabase_router)


@app.get("/")
def _root():
    return Response(content=html, media_type="text/html")


# 在应用程序启动时执行的初始化操作，方式1
app.state.config = AppConfig()


# 在应用程序启动时执行的初始化操作，方式2
@app.on_event("startup")
async def startup_event():
    # 例如，连接到数据库、加载配置、初始化缓存等
    print("print-Initializing data...")
    log.info("log-Initializing data...")


def init_supabase():
    load_dotenv(override=True)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase_token = os.environ.get("SUPABASE_TOKEN")

    print(f"supabase_url: " + supabase_url)
    print(f"supabase_token: " + supabase_token)
    print(f"supabase_key: " + supabase_key)

    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase
