import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.graph import app as graph_app

router = APIRouter()


class ChatRequest(BaseModel):
    text: str = Field(..., description="La pregunta del usuario")
    # Hacemos que el thread_id sea opcional, pero con un valor por defecto dinámico si se prefiere
    thread_id: Optional[str] = Field(
        default=None, description="ID único para mantener la memoria del chat"
    )


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []
    thread_id: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # 1. Dinamismo: Si el cliente no envía un thread_id, generamos uno nuevo.
        # Esto permite que cada nueva pestaña o usuario tenga su propia memoria.
        active_thread_id = request.thread_id or str(uuid.uuid4())

        # 2. Configuración del Grafo para recuperar la memoria correcta
        config = {"configurable": {"thread_id": active_thread_id}}

        # 3. Invocación del grafo
        result = await graph_app.ainvoke({"input": request.text}, config=config)

        return {
            "response": result["response"],
            "sources": result.get("sources", []),
            "thread_id": active_thread_id,  # Importante para que el cliente lo use en la siguiente pregunta
            "history": result["history"],
        }

    except Exception as e:
        print(f"Error en el flujo del grafo: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
