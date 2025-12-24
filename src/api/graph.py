from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.api.core.config import settings
from src.api.core.state import GraphState
from src.api.llm.classifier import classify_intent
from src.api.llm.client import llm
from src.api.search.service import AzureAISearchService

# 1. Configuración de Herramientas
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)

search_service = AzureAISearchService()

# Definimos la herramienta.
# NOTA: Si el error persiste, asegúrate de que en service.py el método esté bien escrito.
search_tool = StructuredTool.from_function(
    coroutine=search_service.search_technical_docs,
    name="search_technical_docs",
    description="Busca información técnica sobre Azure (SQL, App Services, Redes, etc.)",
)

tools_map = {"search_technical_docs": search_tool}
llm_with_tools = llm.bind_tools([search_tool])

# --- NODOS ---


async def agent_node(state: GraphState):
    """Nodo principal que clasifica y decide el siguiente paso."""
    # Solo clasificamos si no lo hemos hecho en este paso del thread
    intent_response = await classify_intent(state["input"])
    intent = intent_response.intention

    if intent == "FUERA_DE_DOMINIO":
        return {"intention": "FUERA_DE_DOMINIO"}

    messages = state.get("history", [])

    # Si es el inicio de la conversación, agregamos el System Message
    if not messages:
        messages = [
            SystemMessage(
                content=(
                    "Eres un asistente experto en Azure. SIEMPRE utiliza la herramienta "
                    "'search_technical_docs' para obtener datos precisos. "
                    "Responde de forma profesional y cita tus fuentes."
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    else:
        # Si ya hay historia, solo agregamos el nuevo input si el último no fue del humano
        if not isinstance(messages[-1], HumanMessage):
            messages.append(HumanMessage(content=state["input"]))

    response = await llm_with_tools.ainvoke(messages)

    # Si hay tool_calls, informamos que estamos buscando
    display_response = (
        response.content
        if not response.tool_calls
        else "Buscando en los manuales de Azure..."
    )

    return {"history": [response], "response": display_response, "intention": intent}


async def out_of_domain_node(state: GraphState):
    """Nodo de seguridad: Rechazo amigable."""
    response = (
        "Lo siento, soy un asistente especializado en Azure. "
        "No puedo ayudarte con temas fuera de la plataforma técnica. "
        "¿Tienes alguna duda sobre VNets, SQL o App Services?"
    )
    return {"response": response}


async def manual_tool_node(state: GraphState):
    """Ejecuta las herramientas y captura fuentes/scores."""
    last_message = state["history"][-1]
    tool_outputs = []
    all_sources = state.get("sources", [])

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "search_technical_docs":
                # La herramienta ya trae el Re-Ranking (Threshold) aplicado desde el service
                observation = await tools_map["search_technical_docs"].ainvoke(
                    tool_call["args"]
                )

                content = observation.get("content", "")
                sources = observation.get("sources", [])

                tool_outputs.append(
                    ToolMessage(content=content, tool_call_id=tool_call["id"])
                )
                all_sources.extend(sources)

    return {"history": tool_outputs, "sources": list(set(all_sources))}


def should_continue(state: GraphState):
    """Router lógico del grafo."""
    if state.get("intention") == "FUERA_DE_DOMINIO":
        return "out_of_domain"

    last_message = state["history"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


# --- CONSTRUCCIÓN DEL WORKFLOW ---
workflow = StateGraph(GraphState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", manual_tool_node)
workflow.add_node("out_of_domain", out_of_domain_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "out_of_domain": "out_of_domain",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("out_of_domain", END)

# MEJORA: Persistencia con MemorySaver
# Esto permite que el thread_id realmente mantenga la memoria
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
