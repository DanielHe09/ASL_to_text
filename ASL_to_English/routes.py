from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import json

router = APIRouter()

class Note(BaseModel):
    id: int
    title: str
    content: str

@router.get("/health")
async def health():
    return {"status": "ok"}