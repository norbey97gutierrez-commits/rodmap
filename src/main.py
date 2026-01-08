"""
MAIN: APLICACIÓN FASTAPI PRINCIPAL
Punto de entrada del backend. Configura la API y middleware CORS.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as chat_router

# CONFIGURACIÓN DE LA APLICACIÓN FASTAPI
app = FastAPI(
    title="Azure AI Architect Backend",
    description="Backend RAG con Azure AI Search, LangGraph y Structured Output",
    version="1.0.0",
)

# MIDDLEWARE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INCLUSIÓN DE RUTAS
app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Azure AI Architect API is running", "docs": "/docs"}


# PUNTO DE ENTRADA PARA EJECUCIÓN LOCAL
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
