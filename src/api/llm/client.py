import logging

from langchain_openai import AzureChatOpenAI

from src.api.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DEL CLIENTE AZURE OPENAI
# ============================================================================

try:
    llm = AzureChatOpenAI(
        azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        # --- PARÁMETROS DE COMPORTAMIENTO ---
        temperature=0.0,  # Precisión técnica determinística
        streaming=True,  # Habilita el envío de tokens en tiempo real
        max_retries=3,  # Reintentos automáticos en errores 429 o 500
        timeout=60.0,  # Evita peticiones colgadas (en segundos)
        # --- SEGURIDAD ---
        # verbose=True,        # Útil solo en desarrollo local
    )
    logger.info("✅ Cliente AzureChatOpenAI inicializado correctamente.")

except Exception as e:
    logger.critical(f"❌ Error al inicializar AzureChatOpenAI: {str(e)}")
    raise e
