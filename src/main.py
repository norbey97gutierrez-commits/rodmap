"""
MAIN: APLICACIÓN FASTAPI PRINCIPAL
Punto de entrada minimal.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.adapters.local.postgres_repo import PostgresRepo
from src.infrastructure.azure_setup import settings
from src.routes import router

PostgresRepo()

app = FastAPI(
    title="Azure AI Architect Backend",
    description="Backend RAG con Azure AI Search, LangGraph y Structured Output",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENV == "dev",
    )
