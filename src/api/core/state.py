from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ============================================================================
# FUNCIONES DE FUSIÓN (REDUCERS)
# ============================================================================


def merge_sources(left: Optional[List[str]], right: Optional[List[str]]) -> List[str]:
    """
    Combina fuentes de manera única preservando el orden.
    Annotated indica a LangGraph que use esta función para 'sumar' estados.
    """
    if not left:
        left = []
    if not right:
        right = []

    # Combinación eficiente manteniendo orden de inserción (Python 3.7+)
    return list(dict.fromkeys(left + right))


# ============================================================================
# DEFINICIÓN DEL ESTADO DEL GRAFO
# ============================================================================


class GraphState(TypedDict):
    """
    Esquema de estado para el orquestador de Azure OpenAI.
    """

    # ENTRADA: Inmutable durante el ciclo del grafo
    input: str

    # CLASIFICACIÓN: Controla el flujo en 'should_continue'
    # Valores: "EN_DOMINIO", "FUERA_DE_DOMINIO"
    intention: Optional[str]

    # FUENTES: Acumulativas y únicas mediante merge_sources
    sources: Annotated[List[str], merge_sources]

    # RESPUESTA: El texto final o parcial para el usuario
    response: str

    # MEMORIA: Gestionada automáticamente por LangGraph (add_messages)
    # Fundamental para que el Checkpointer persista la sesión.
    history: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_initial_state(user_input: str) -> GraphState:
    """Crea la estructura base para una nueva ejecución."""
    return {
        "input": user_input,
        "intention": None,
        "sources": [],
        "response": "",
        "history": [],
    }
