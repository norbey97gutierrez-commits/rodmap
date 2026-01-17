from enum import Enum

from pydantic import BaseModel, Field


class SearchTechnicalDocsInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "La pregunta o consulta del usuario sobre Azure "
            "(VNets, SQL, escalabilidad, etc.)"
        ),
    )


class IntentionEnum(str, Enum):
    SALUDO = "SALUDO"
    PREGUNTA_TECNICA = "PREGUNTA_TECNICA"
    FUERA_DE_DOMINIO = "FUERA_DE_DOMINIO"


class IntentionResponse(BaseModel):
    """Esquema de salida forzado para el LLM."""

    intention: IntentionEnum = Field(..., description="La categoría de la pregunta.")
    reasoning: str = Field(
        ..., description="Breve explicación de por qué se eligió esta categoría."
    )
