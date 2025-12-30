import json
import logging
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Importamos la instancia del grafo
from src.api.graph import app as graph_app

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

# --- MODELOS DE DATOS ---


class ChatRequest(BaseModel):
    # Validación: No permitimos textos vacíos para ahorrar tokens y latencia
    text: str = Field(
        ..., min_length=1, max_length=4000, example="¿Cómo configurar una VNet?"
    )
    thread_id: Optional[str] = Field(None, example="sesion-123")


# --- UTILIDADES DE STREAMING ---


async def stream_graph_updates(text: str, thread_id: str) -> AsyncGenerator[str, None]:
    """
    Generador para transmitir la respuesta del grafo en tiempo real.
    Maneja el formateo de eventos para que el frontend pueda procesarlos.
    """
    inputs = {"input": text}
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Usamos astream con modo "messages" o "values" según tu configuración de grafo
        # 'astream_events' es lo más potente para capturar streaming de Azure OpenAI
        async for event in graph_app.astream_events(inputs, config, version="v2"):
            kind = event["event"]

            # Evento: El LLM está generando texto (streaming de tokens)
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Formato Server-Sent Events (SSE)
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            # Evento: El grafo terminó un nodo (útil para fuentes/sources)
            elif kind == "on_chain_end":
                if event["name"] == "LangGraph":  # Nombre raíz del grafo
                    data = event["data"]["output"]
                    sources = data.get("sources", [])
                    yield f"data: {json.dumps({'type': 'metadata', 'sources': sources})}\n\n"

    except Exception as e:
        logger.error(f"Error en stream: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Error en la generación'})}\n\n"


# --- ENDPOINTS ---


@router.post("/stream")
async def chat_endpoint_stream(request: ChatRequest):
    """
    Endpoint de Chat con Streaming.
    Ideal para aplicaciones de React que usan hooks como 'useChat' o EventSource.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        stream_graph_updates(request.text, thread_id), media_type="text/event-stream"
    )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "azure-ai-chat-api"}
