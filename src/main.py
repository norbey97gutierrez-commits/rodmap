"""
MAIN: APLICACIÓN FASTAPI PRINCIPAL
Punto de entrada del backend. Configura la API y middleware CORS.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN FASTAPI
# ============================================================================
app = FastAPI(
    title="Azure AI Roadmap",
    description="Backend para RAG con Azure AI Search y LangGraph",
    version="1.0.0",
    # Opciones adicionales (comentadas para referencia):
    # docs_url="/docs",      # Habilitar Swagger UI
    # redoc_url="/redoc",    # Habilitar ReDoc
    # openapi_url="/openapi.json"
)

# ============================================================================
# MIDDLEWARE CORS (CROSS-ORIGIN RESOURCE SHARING)
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (ajustar en producción)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)
"""
IMPORTANTE: En producción, reemplazar "*" con los dominios específicos del frontend.
Ejemplo: allow_origins=["https://tudominio.com", "https://app.tudominio.com"]
"""

# ============================================================================
# INCLUSIÓN DE RUTAS
# ============================================================================
app.include_router(router, prefix="/api/v1")

# ============================================================================
# PUNTO DE ENTRADA PARA EJECUCIÓN LOCAL
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    # Configuración del servidor de desarrollo
    uvicorn.run(
        "src.api.main:app",  # Módulo de la aplicación
        host="0.0.0.0",  # Escuchar en todas las interfaces
        port=8000,  # Puerto por defecto
        reload=True,  # Recarga automática en cambios de código
    )
