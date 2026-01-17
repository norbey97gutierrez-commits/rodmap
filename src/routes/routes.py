import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.application.graph import build_graph
from src.routes.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@asynccontextmanager
async def _sqlite_saver():
    async with aiosqlite.connect("data/dev_history.db") as conn:
        if not hasattr(conn, "is_alive"):
            conn.is_alive = lambda: True  # type: ignore[attr-defined]
        saver = AsyncSqliteSaver(conn)
        yield saver


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    intention: str
    response: str
    sources: List[Any]
    status: str


@router.post("/chat/query", response_model=ChatResponse)
async def chat_endpoint_json(request: ChatRequest, _user=Depends(get_current_user)):
    thread_id = request.thread_id or str(uuid.uuid4())

    inputs = {"input": request.text}
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"Invocando grafo para thread_id: {thread_id}")
        logger.info(f"Input: '{request.text[:100]}...'")

        async with _sqlite_saver() as saver:
            graph_app = build_graph(saver)
            if request.thread_id:
                try:
                    current_state = await graph_app.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    if current_state and current_state.values:
                        last_input = current_state.values.get("input", "")
                        if last_input and last_input.strip() != request.text.strip():
                            old_thread_id = thread_id
                            thread_id = str(uuid.uuid4())
                            logger.warning(
                                "*** INPUT DIFERENTE DETECTADO - GENERANDO NUEVO THREAD_ID ***"
                            )
                            logger.warning(f"Input anterior: '{last_input[:50]}...'")
                            logger.warning(f"Input nuevo: '{request.text[:50]}...'")
                            logger.warning(f"Thread ID cambiado: {old_thread_id} -> {thread_id}")
                        else:
                            logger.info(
                                f"Mismo input detectado, manteniendo thread_id: {thread_id}"
                            )
                    else:
                        logger.info(
                            f"No hay estado previo para thread_id: {thread_id}, nueva consulta"
                        )
                except Exception as e:
                    logger.debug(f"No se pudo obtener estado previo (nueva consulta): {e}")
            else:
                logger.info(f"Sin thread_id del frontend, generando nuevo: {thread_id}")

            final_state = await graph_app.ainvoke(inputs, config)

        response_text = final_state.get("response", "") if final_state else ""
        if not response_text or not response_text.strip():
            response_text = (
                "No pude generar una respuesta para tu pregunta. Por favor, intenta reformularla."
            )
            logger.warning("Respuesta vacía, usando mensaje por defecto")

        sources = final_state.get("sources", []) if final_state else []
        intention = str(final_state.get("intention", "UNKNOWN")) if final_state else "UNKNOWN"

        return ChatResponse(
            thread_id=thread_id,
            intention=intention,
            response=response_text,
            sources=sources,
            status="success",
        )

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_type = type(e).__name__
        error_message = str(e)

        logger.error("=" * 80)
        logger.error("ERROR EN CHAT_ENDPOINT")
        logger.error(f"Tipo: {error_type}")
        logger.error(f"Mensaje: {error_message}")
        logger.error("Traceback completo:")
        logger.error(error_traceback)
        logger.error("=" * 80)

        if (
            "tool" in error_message.lower()
            or "tool_calls" in error_message.lower()
            or "tool_calls" in error_type.lower()
        ):
            user_message = (
                "Problema con la ejecución de herramientas de la IA. "
                "Por favor, intenta reformular tu pregunta o contacta al administrador."
            )
        elif "timeout" in error_message.lower():
            user_message = (
                "La solicitud tardó demasiado tiempo. "
                "Por favor, intenta con una pregunta más específica."
            )
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            user_message = (
                "Error de conexión con los servicios de Azure. "
                "Por favor, intenta nuevamente en unos momentos."
            )
        else:
            user_message = "Error interno del servidor. Por favor, intenta reformular tu pregunta."

        raise HTTPException(status_code=500, detail=user_message)


@router.get("/chat/health")
async def health_check(_user=Depends(get_current_user)):
    return {"status": "healthy"}


@router.get("/")
async def root():
    return {"message": "Azure AI Architect API is running", "docs": "/docs"}
