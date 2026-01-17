import logging

from langchain_openai import AzureChatOpenAI

from src.infrastructure.azure_setup import settings

logger = logging.getLogger(__name__)


def build_chat_client() -> AzureChatOpenAI:
    try:
        return AzureChatOpenAI(
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.0,
            streaming=False,
            max_retries=3,
            timeout=60.0,
        )
    except Exception as e:
        logger.critical(f"Error al inicializar AzureChatOpenAI: {str(e)}")
        raise e


llm = build_chat_client()
