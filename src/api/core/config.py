# src/api/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # =======================
    # Azure OpenAI (Modelos)
    # =======================
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-05-01-preview"

    # Nombre del despliegue para GPT-4 (Chat)
    AZURE_OPENAI_CHAT_DEPLOYMENT: str

    # Nombre del despliegue para Embeddings (el de 3072 dimensiones)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str

    # =======================
    # Azure AI Search
    # =======================
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_API_KEY: str
    AZURE_SEARCH_INDEX_NAME: str

    # Configuración de Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Cambiado a ignore para evitar que falle si hay variables extra en el .env
    )


settings = Settings()
