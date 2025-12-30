import logging
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.api.llm.client import llm

logger = logging.getLogger(__name__)

# ============================================================================
# MODELOS DE DATOS
# ============================================================================


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


# Configuración del LLM para usar Salida Estructurada (Native JSON Mode/Tools)
# Esto hace que el LLM se comporte como un validador de Pydantic
structured_llm = llm.with_structured_output(IntentionResponse)

# ============================================================================
# CLASIFICADOR PRINCIPAL
# ============================================================================


async def classify_intent(text: str) -> IntentionResponse:
    """
    Clasifica la intención usando Structured Output de Azure OpenAI.
    Elimina la necesidad de Regex y limpieza manual.
    """

    system_instruction = (
        "Eres un clasificador experto para un asistente de Microsoft Azure.\n"
        "Categoriza la entrada según estas reglas:\n"
        "1. SALUDO: Cortesías y charlas breves.\n"
        "2. PREGUNTA_TECNICA: Dudas sobre servicios de Azure (VNet, SQL, App Service, etc).\n"
        "3. FUERA_DE_DOMINIO: Temas no relacionados con Azure o tecnología.\n"
        "\n"
        "IMPORTANTE: Si mencionan AWS o Google Cloud, clasifica como FUERA_DE_DOMINIO."
    )

    try:
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Entrada del usuario: '{text}'"),
        ]

        # Invocación directa: devuelve una instancia de IntentionResponse, no un string
        result = await structured_llm.ainvoke(messages)

        # Log para trazabilidad en Azure Monitor
        logger.info(f"Clasificación: {result.intention} | Razón: {result.reasoning}")

        return result

    except Exception as e:
        logger.error(f"Error crítico en clasificación: {str(e)}", exc_info=True)

        # Fallback seguro: Enrutamos a técnica para que el RAG intente responder
        return IntentionResponse(
            intention=IntentionEnum.PREGUNTA_TECNICA,
            reasoning="Fallback por error en el servicio de clasificación.",
        )
