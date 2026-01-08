from typing import Annotated, Any, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_sources(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    """
    Mantiene una lista única de fuentes basada en el título y página.
    """
    if not left:
        left = []
    if not right:
        right = []

    combined = left + right
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


class GraphState(TypedDict):
    input: str
    intention: Optional[str]
    context: str
    sources: Annotated[List[Any], merge_sources]
    response: str
    history: Annotated[Sequence[BaseMessage], add_messages]
