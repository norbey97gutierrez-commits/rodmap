import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


async def extractor_node(state: dict) -> dict:
    """
    Nodo de finalización que extrae la respuesta final y las fuentes.
    """
    history = state.get("history", [])

    last_ai_msg = next(
        (
            m
            for m in reversed(history)
            if isinstance(m, AIMessage) and (not m.tool_calls or len(m.tool_calls) == 0)
        ),
        None,
    )

    response_text = last_ai_msg.content if last_ai_msg else ""

    if not response_text or not response_text.strip():
        response_text = (
            "No pude generar una respuesta para tu pregunta. Por favor, intenta reformularla."
        )
        logger.warning("extractor_node - No se encontró respuesta, usando mensaje por defecto")

    current_response_sources = []
    seen_keys = set()

    for msg in reversed(history):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                docs = data.get("value", []) if isinstance(data, dict) else []

                for doc in docs:
                    source_name = doc.get("source") or doc.get("title") or ""
                    clean_name = (
                        str(source_name)
                        .split("\\")[-1]
                        .split("/")[-1]
                        .replace(".pdf", "")
                        .replace(".docx", "")
                    )

                    if clean_name.lower() in response_text.lower():
                        source_key = f"{clean_name}-{doc.get('page_number')}"
                        if source_key not in seen_keys:
                            seen_keys.add(source_key)
                            current_response_sources.append(
                                {
                                    "title": f"{clean_name}.pdf",
                                    "page": doc.get("page_number"),
                                    "url": doc.get("url") or "#",
                                }
                            )
            except Exception as e:
                logger.debug(f"extractor_node - Error extrayendo fuente: {e}")
                continue

        if isinstance(msg, HumanMessage):
            break

    return {
        "response": response_text,
        "sources": current_response_sources,
        "history": history,
    }
