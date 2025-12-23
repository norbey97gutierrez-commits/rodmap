from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# Función para acumular fuentes sin duplicados
def merge_sources(left: List[str], right: List[str]) -> List[str]:
    return list(set((left or []) + (right or [])))


class GraphState(TypedDict):
    input: str
    intention: str
    context: str
    # Usamos Annotated para que LangGraph sepa cómo mezclar las fuentes
    sources: Annotated[List[str], merge_sources]
    response: str
    history: Annotated[Sequence[BaseMessage], add_messages]
