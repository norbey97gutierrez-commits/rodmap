from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .llm.client import get_chat_response

router = APIRouter()


class ChatRequest(BaseModel):
    text: str


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await get_chat_response(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
