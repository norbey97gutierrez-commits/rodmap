from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import (
    ToolNode,  # Nodo pre-construido para ejecutar herramientas
)

from src.api.core.config import settings
from src.api.core.state import GraphState
from src.api.llm.client import llm
from src.api.search.service import AzureAISearchService

# 1. Configuración de Servicios y Herramientas
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)

search_service = AzureAISearchService(
    endpoint=settings.AZURE_SEARCH_ENDPOINT, api_key=settings.AZURE_SEARCH_API_KEY
)

# Definimos las herramientas que el agente puede usar
tools = [search_service.search_technical_docs]
# Vinculamos las herramientas al modelo (Tool Binding)
llm_with_tools = llm.bind_tools(tools)

# --- NODOS DEL GRAFO ---


async def agent_node(state: GraphState):
    """
    Nodo del Agente: Decide si usar una herramienta o responder directamente.
    """
    existing_history = state.get("history") or []

    system_msg = SystemMessage(
        content=(
            "Eres un asistente técnico experto en Azure. "
            "Si necesitas datos específicos sobre servicios, usa la herramienta de búsqueda."
        )
    )

    messages = [system_msg] + existing_history + [HumanMessage(content=state["input"])]

    # El LLM ahora puede devolver una respuesta de texto o una solicitud de herramienta
    response = await llm_with_tools.ainvoke(messages)

    return {
        "history": [HumanMessage(content=state["input"]), response],
        "response": response.content,
    }


def should_continue(state: GraphState):
    """
    Lógica de control: Revisa si el último mensaje del historial es una llamada a herramienta.
    """
    messages = state.get("history", [])
    last_message = messages[-1] if messages else None

    if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # Ir al nodo de ejecución de herramientas
    return END  # Finalizar si no hay herramientas que ejecutar


# --- CONSTRUCCIÓN DEL WORKFLOW ---
workflow = StateGraph(GraphState)

# Añadimos el nodo del agente
workflow.add_node("agent", agent_node)

# Añadimos el ToolNode (ejecuta automáticamente las herramientas vinculadas)
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

# Añadimos el borde condicional para el ciclo de razonamiento (Agentic Loop)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    },
)

# El resultado de la herramienta siempre vuelve al agente para que lo analice
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
