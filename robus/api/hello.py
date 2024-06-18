#!/usr/bin/env python
from fastapi.routing import APIRouter

router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "World"}


@router.get("/v1")
def read_root():
    return {"Hello": "test"}
