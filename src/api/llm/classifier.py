import re
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import AliasChoices, BaseModel, Field

from src.api.llm.client import llm


class IntentionEnum(str, Enum):
    SALUDO = "SALUDO"
    PREGUNTA_TECNICA = "PREGUNTA_TECNICA"
    FUERA_DE_DOMINIO = "FUERA_DE_DOMINIO"


class IntentionResponse(BaseModel):
    # AliasChoices permite que funcione con 'intention' O 'intencion'
    intention: IntentionEnum = Field(
        validation_alias=AliasChoices("intention", "intencion")
    )


def extract_json(text: str) -> str:
    """Extrae el primer objeto JSON válido de un texto."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return '{"intention": "FUERA_DE_DOMINIO"}'
    return match.group(0)


async def classify_intent(text: str) -> IntentionResponse:
    system_instruction = (
        "Eres un clasificador de intenciones experto. Categoriza la entrada en: "
        "SALUDO, PREGUNTA_TECNICA, o FUERA_DE_DOMINIO.\n"
        'Responde con este formato JSON: {"intention": "CATEGORIA"}'
    )

    try:
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Clasifica: '{text}'"),
        ]

        response = await llm.ainvoke(messages)

        # Eliminamos la importación interna que causaba el error 'cannot import name'
        json_str = extract_json(response.content)

        return IntentionResponse.model_validate_json(json_str)

    except Exception as e:
        print(f"Error detectado en clasificador: {e}")
        # Fallback seguro
        return IntentionResponse(intention=IntentionEnum.PREGUNTA_TECNICA)
