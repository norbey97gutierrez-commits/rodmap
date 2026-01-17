from typing import Annotated, Any, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages


def merge_sources(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    """
    Mantiene una lista única de fuentes basada en el título y página.
    """
    if right is None:
        return list(left) if left else []

    # Siempre reemplazamos fuentes por las del último turno
    return list(right)
    unique_sources = []
    seen_ids = set()

    for s in combined:
        if isinstance(s, dict):
            s_id = f"{s.get('title', '')}-{s.get('url', '')}"
        else:
            s_id = str(s)

        if s_id not in seen_ids:
            seen_ids.add(s_id)
            unique_sources.append(s)

    return unique_sources


def merge_history_with_reset(
    left: Optional[Sequence[BaseMessage]],
    right: Optional[Sequence[BaseMessage]],
) -> Sequence[BaseMessage]:
    """
    Merge personalizado para el historial que detecta cambios en el input
    y reinicia el historial completamente cuando el último mensaje humano cambia.
    """
    import logging

    logger = logging.getLogger(__name__)

    if not left:
        left = []
    if not right:
        right = []

    if not right:
        return list(left) if left else []

    last_human_left = next(
        (m for m in reversed(left) if isinstance(m, HumanMessage)), None
    )

    first_human_right = next((m for m in right if isinstance(m, HumanMessage)), None)

    logger.info(f"merge_history - Left: {len(left)} msgs, Right: {len(right)} msgs")
    if last_human_left:
        logger.info(
            f"merge_history - Último humano en left: '{last_human_left.content[:50]}...'"
        )
    if first_human_right:
        logger.info(
            f"merge_history - Primer humano en right: '{first_human_right.content[:50]}...'"
        )

    if not first_human_right:
        logger.info("merge_history - Right no tiene HumanMessage, agregando normalmente")
        return list(left) + list(right)

    if not last_human_left:
        logger.info("merge_history - Sin historial previo, usando right directamente")
        return list(right)

    left_content = last_human_left.content.strip()
    right_content = first_human_right.content.strip()

    if right_content == left_content:
        logger.info("merge_history - Mismo input, agregando mensajes normalmente")
        return list(left) + list(right)

    has_pending_tool_calls = _has_pending_tool_calls(left)

    if has_pending_tool_calls:
        logger.warning("merge_history - Input diferente PERO hay tool_calls pendientes")
        logger.warning(
            "merge_history - NO reiniciando - agregando normalmente para completar tool_calls"
        )
        return list(left) + list(right)

    logger.warning("merge_history - *** REINICIO COMPLETO DEL HISTORIAL ***")
    logger.warning(
        f"merge_history - Input anterior ({len(left_content)} chars): '{left_content[:50]}...'"
    )
    logger.warning(
        f"merge_history - Input nuevo ({len(right_content)} chars): '{right_content[:50]}...'"
    )
    logger.warning(
        f"merge_history - Historial anterior: {len(left)} msgs → Nuevo: {len(right)} msgs"
    )

    return list(right)


def _has_pending_tool_calls(history: Sequence[BaseMessage]) -> bool:
    if not history:
        return False

    last_ai_with_tools = None
    for msg in reversed(history):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            last_ai_with_tools = msg
            break

    if not last_ai_with_tools:
        return False

    tool_call_ids = {tc.get("id") for tc in last_ai_with_tools.tool_calls if tc.get("id")}
    if not tool_call_ids:
        return False

    tool_message_ids = {
        msg.tool_call_id
        for msg in history
        if isinstance(msg, ToolMessage)
        and hasattr(msg, "tool_call_id")
        and msg.tool_call_id
    }

    pending_ids = tool_call_ids - tool_message_ids
    if pending_ids:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"_has_pending_tool_calls - Tool calls pendientes: {pending_ids}")
        return True

    return False


class GraphState(TypedDict):
    input: str
    intention: Optional[str]
    context: str
    sources: Annotated[List[Any], merge_sources]
    response: str
    history: Annotated[Sequence[BaseMessage], merge_history_with_reset]
