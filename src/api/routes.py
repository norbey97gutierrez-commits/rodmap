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

# --- MODELOS DE DATOS ---


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    intention: str
    response: str
    sources: List[Any]  # Lista de diccionarios {title, url, page}
    status: str


# --- ENDPOINTS ---


@router.post("/query", response_model=ChatResponse)
async def chat_endpoint_json(request: ChatRequest):
    """
    Procesa la pregunta y devuelve la respuesta JSON con fuentes.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    # IMPORTANTE: El input debe coincidir con las llaves de tu GraphState
    # Si tu grafo espera 'input' y 'history', los enviamos aquí.
    inputs = {
        "input": request.text,
        "history": [],  # LangGraph recuperará el historial del checkpointer usando el thread_id
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"Invocando grafo para thread_id: {thread_id}")

        # Ejecución del grafo: ainvoke devuelve el estado FINAL tras pasar por 'finalize'
        final_state = await graph_app.ainvoke(inputs, config)

        # DEBUG: Verifica en consola si el estado final contiene lo que finalize_node generó
        logger.info(f"Estado final recuperado: {final_state.keys()}")

        # Extraemos los campos que finalize_node inyectó en el estado
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
