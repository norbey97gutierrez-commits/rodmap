from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="Azure AI Roadmap",
    description="Backend profesional para orquestación de RAG con Azure AI Search y LangGraph",
    version="1.0.0",
)

# Configuración de CORS (Vital si vas a conectar un Frontend como React o Vue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia "*" por tus dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusión de rutas con versionamiento
app.include_router(router, prefix="/api/v1")

# Punto de entrada para ejecución local
if __name__ == "__main__":
    import uvicorn

    # Log_level "info" para ver las peticiones en la consola
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
