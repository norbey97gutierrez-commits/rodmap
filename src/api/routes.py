import logging
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.graph import app as graph_app

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


# MODELOS DE DATOS
class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    intention: str
    response: str
    sources: List[Any]
    status: str


# ENDPOINTS
@router.post("/query", response_model=ChatResponse)
async def chat_endpoint_json(request: ChatRequest):
    """
    Procesa la pregunta y devuelve la respuesta JSON con fuentes.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    inputs = {
        "input": request.text,
        "history": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"Invocando grafo para thread_id: {thread_id}")
        final_state = await graph_app.ainvoke(inputs, config)
        logger.info(f"Estado final recuperado: {final_state.keys()}")
        response_text = final_state.get(
            "response", "No pude generar una respuesta técnica."
        )
        sources = final_state.get("sources", [])
        intention = str(final_state.get("intention", "UNKNOWN"))

        logger.info(f"Fuentes encontradas: {len(sources)}")

        return ChatResponse(
            thread_id=thread_id,
            intention=intention,
            response=response_text,
            sources=sources,
            status="success",
        )

    except Exception as e:
        logger.error(f"Error en chat_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
