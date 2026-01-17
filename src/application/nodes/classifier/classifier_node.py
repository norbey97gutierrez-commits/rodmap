import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.adapters.azure.openai_client import llm
from src.adapters.parsers.output_parsers import build_structured_llm
from src.domain.entities.schemas import IntentionEnum, IntentionResponse

logger = logging.getLogger(__name__)

structured_llm = build_structured_llm(llm)


async def classify_intent(text: str) -> IntentionResponse:
    system_instruction = (
        "Eres un clasificador experto para un asistente de Microsoft Azure.\n"
        "Categoriza la entrada según estas reglas:\n"
        "1. SALUDO: Cortesías y charlas breves.\n"
        "2. PREGUNTA_TECNICA: Dudas sobre servicios de Azure (VNet, SQL, App Service, etc).\n"
        "3. FUERA_DE_DOMINIO: Temas no relacionados con Azure o tecnología.\n"
        "\n"
        "IMPORTANTE: Si la pregunta menciona AWS, Amazon Web Services, Google Cloud, GCP "
        "o cualquier tecnología que no sea Microsoft Azure, clasifica OBLIGATORIAMENTE "
        "como FUERA_DE_DOMINIO."
    )

    try:
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Entrada del usuario: '{text}'"),
        ]

        result = await structured_llm.ainvoke(messages)
        logger.info(f"Clasificación: {result.intention} | Razón: {result.reasoning}")
        return result

    except Exception as e:
        logger.error(f"Error crítico en clasificación: {str(e)}", exc_info=True)
        return IntentionResponse(
            intention=IntentionEnum.PREGUNTA_TECNICA,
            reasoning="Fallback por error en el servicio de clasificación.",
        )


async def classifier_node(state: dict) -> dict:
    current_input = state.get("input", "")
    intent_response = await classify_intent(current_input)
    return {"intention": intent_response.intention}
