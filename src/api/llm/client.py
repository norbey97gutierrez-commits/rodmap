import logging

from langchain_openai import AzureChatOpenAI

from src.api.core.config import settings

logger = logging.getLogger(__name__)


# CONFIGURACIÓN DEL CLIENTE AZURE OPENAI
try:
    llm = AzureChatOpenAI(
        azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=0.1,
        streaming=True,
        max_retries=3,
        timeout=60.0,
    )
    logger.info("Cliente AzureChatOpenAI inicializado correctamente.")

except Exception as e:
    logger.critical(f"Error al inicializar AzureChatOpenAI: {str(e)}")
    raise e
