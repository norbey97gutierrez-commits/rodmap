import logging
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Configuración centralizada con validación estricta de Azure."""

    # APP SETTINGS
    ENV: Literal["dev", "prod", "test"] = "dev"
    LOG_LEVEL: str = "INFO"

    # AZURE OPENAI
    AZURE_OPENAI_ENDPOINT: HttpUrl = Field(
        ..., example="https://resource.openai.azure.com/"
    )
    AZURE_OPENAI_API_KEY: str = Field(..., min_length=32)
    AZURE_OPENAI_API_VERSION: str = "2024-05-01-preview"

    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(...)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(...)

    # AZURE AI SEARCH
    AZURE_SEARCH_ENDPOINT: HttpUrl = Field(...)
    AZURE_SEARCH_API_KEY: str = Field(...)
    AZURE_SEARCH_INDEX_NAME: str = Field(...)

    # VALIDACIONES NATIVAS
    @field_validator("AZURE_OPENAI_API_KEY", "AZURE_SEARCH_API_KEY")
    @classmethod
    def check_empty_keys(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Las API Keys no pueden ser espacios en blanco")
        return v

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )


# Instancia Singleton
settings = Settings()

# Configuración básica de logging inmediata
logging.basicConfig(level=settings.LOG_LEVEL)
logger.info(f"Configuración cargada en modo: {settings.ENV}")
