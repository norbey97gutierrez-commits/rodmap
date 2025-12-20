from langchain_openai import AzureChatOpenAI

from src.api.core.config import settings

# Inicialización del modelo de lenguaje (GPT-4)
llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    # Sincronizado con el nuevo nombre en settings para el modelo de chat
    azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
    temperature=0.0,  # 0.0 para máxima precisión técnica y evitar alucinaciones
)
