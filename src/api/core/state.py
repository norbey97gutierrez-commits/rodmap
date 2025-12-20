import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    input: str
    intention: str
    context: str
    sources: List[str]
    response: str
    # El historial ahora es una lista de mensajes que se van acumulando
    history: Annotated[List[BaseMessage], operator.add]
