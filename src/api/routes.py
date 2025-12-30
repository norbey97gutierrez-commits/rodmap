import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Importamos la instancia del grafo compilado
from src.api.graph import app as graph_app

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

# ============================================================================
# MODELOS DE DATOS
# ============================================================================


class ChatRequest(BaseModel):
    """Esquema de entrada para la consulta."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="La pregunta técnica sobre Azure.",
        example="¿Cómo se configura una VNet?",
    )
    thread_id: Optional[str] = Field(
        None,
        description="ID de sesión para mantener el historial de la conversación.",
        example="sesion-azure-001",
    )


class ChatResponse(BaseModel):
    """Esquema de salida JSON único."""

    thread_id: str
    intention: str
    response: str
    sources: list[str]
    status: str


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/stream", response_model=ChatResponse)
async def chat_endpoint_json(request: ChatRequest):
    """
    Endpoint que procesa la pregunta y devuelve una respuesta JSON completa.
    Ejecuta el flujo RAG: Clasificación -> Búsqueda -> Generación -> Citas.
    """
    # Generar un thread_id si no se proporciona para mantener la trazabilidad
    thread_id = request.thread_id or str(uuid.uuid4())

    # Preparar la entrada para el Grafo de LangGraph
    inputs = {"input": request.text}
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"Procesando consulta para thread_id: {thread_id}")

        # Invocamos el grafo de forma asíncrona y esperamos el estado final
        # ainvoke recorre todos los nodos (agent -> tools -> agent) automáticamente
        final_state = await graph_app.ainvoke(inputs, config)

        # Extraemos la información del estado final del grafo
        # El intention viene del Enum, lo convertimos a string para el JSON
        return ChatResponse(
            thread_id=thread_id,
            intention=str(final_state.get("intention")),
            response=final_state.get("response", "Sin respuesta generada"),
            sources=final_state.get("sources", []),
            status="success",
        )

    except Exception as e:
        logger.error(f"Error crítico en el endpoint de chat: {str(e)}", exc_info=True)
        # Devolvemos un error 500 estructurado si algo falla en el proceso
        raise HTTPException(
            status_code=500, detail=f"Error interno al procesar la solicitud: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Endpoint simple para verificar que el servicio de rutas está activo."""
    return {"status": "healthy", "service": "azure-ai-chat-api"}
