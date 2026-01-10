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
    # SOLUCIÓN CRÍTICA: Siempre generamos un nuevo thread_id para cada consulta nueva
    # Esto asegura que cada pregunta empiece con un historial limpio
    # Si el frontend quiere mantener continuidad, debe enviar explícitamente el mismo thread_id
    # y nosotros detectaremos si el input cambió para reiniciar el historial
    
    # Si el frontend NO envía thread_id, generamos uno nuevo (cada consulta nueva empieza limpia)
    # Si el frontend SÍ envía thread_id, verificamos si el input cambió
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Verificamos si el input cambió comparando con el estado guardado del checkpointer
    # Si cambió, generamos un nuevo thread_id para forzar reinicio completo
    if request.thread_id:
        # Solo verificamos si el frontend envió un thread_id explícito
        try:
            current_state = await graph_app.aget_state({"configurable": {"thread_id": thread_id}})
            if current_state and current_state.values:
                last_input = current_state.values.get("input", "")
                if last_input and last_input.strip() != request.text.strip():
                    # Input diferente detectado - generamos nuevo thread_id para reinicio completo
                    old_thread_id = thread_id
                    thread_id = str(uuid.uuid4())
                    logger.warning(f"*** INPUT DIFERENTE DETECTADO - GENERANDO NUEVO THREAD_ID ***")
                    logger.warning(f"Input anterior: '{last_input[:50]}...'")
                    logger.warning(f"Input nuevo: '{request.text[:50]}...'")
                    logger.warning(f"Thread ID cambiado: {old_thread_id} -> {thread_id}")
                else:
                    logger.info(f"Mismo input detectado, manteniendo thread_id: {thread_id}")
            else:
                logger.info(f"No hay estado previo para thread_id: {thread_id}, nueva consulta")
        except Exception as e:
            # Si hay error obteniendo el estado (p.ej. thread_id no existe), es una nueva consulta
            logger.debug(f"No se pudo obtener estado previo (nueva consulta): {e}")
    else:
        # No hay thread_id del frontend, cada consulta es nueva
        logger.info(f"Sin thread_id del frontend, generando nuevo: {thread_id}")
    
    inputs = {
        "input": request.text,
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"Invocando grafo para thread_id: {thread_id}")
        logger.info(f"Input: '{request.text[:100]}...'")
        
        # Ejecutamos el grafo con manejo robusto de errores
        final_state = await graph_app.ainvoke(inputs, config)
        
        logger.info(f"Estado final recuperado: {final_state.keys() if final_state else 'None'}")
        
        # Extraemos la respuesta con validaciones
        response_text = final_state.get("response", "") if final_state else ""
        if not response_text or not response_text.strip():
            response_text = "No pude generar una respuesta para tu pregunta. Por favor, intenta reformularla."
            logger.warning(f"Respuesta vacía, usando mensaje por defecto")
        
        sources = final_state.get("sources", []) if final_state else []
        intention = str(final_state.get("intention", "UNKNOWN")) if final_state else "UNKNOWN"

        logger.info(f"Respuesta generada: {len(response_text)} chars, {len(sources)} fuentes, intención: {intention}")

        return ChatResponse(
            thread_id=thread_id,
            intention=intention,
            response=response_text,
            sources=sources,
            status="success",
        )

    except Exception as e:
        # Logging detallado del error
        import traceback
        error_traceback = traceback.format_exc()
        error_type = type(e).__name__
        error_message = str(e)
        
        logger.error(f"=" * 80)
        logger.error(f"ERROR EN CHAT_ENDPOINT")
        logger.error(f"Tipo: {error_type}")
        logger.error(f"Mensaje: {error_message}")
        logger.error(f"Traceback completo:")
        logger.error(error_traceback)
        logger.error(f"=" * 80)
        
        # Mensaje amigable para el usuario
        if "tool" in error_message.lower() or "tool_calls" in error_message.lower() or "tool_calls" in error_type.lower():
            user_message = "Problema con la ejecución de herramientas de la IA. Por favor, intenta reformular tu pregunta o contacta al administrador."
        elif "timeout" in error_message.lower():
            user_message = "La solicitud tardó demasiado tiempo. Por favor, intenta con una pregunta más específica."
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            user_message = "Error de conexión con los servicios de Azure. Por favor, intenta nuevamente en unos momentos."
        else:
            # Mensaje genérico pero informativo
            user_message = f"Error interno del servidor. Por favor, intenta reformular tu pregunta."
        
        raise HTTPException(status_code=500, detail=user_message)


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
