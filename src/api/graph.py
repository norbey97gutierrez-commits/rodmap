import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.api.core.config import settings
from src.api.core.state import GraphState
from src.api.llm.classifier import classify_intent

# Asumimos que llm ya viene configurado con streaming=True desde client.py
from src.api.llm.client import llm
from src.api.search.service import AzureAISearchService

logger = logging.getLogger(__name__)

# ============================================================================
# 1. CONFIGURACIÓN DE COMPONENTES
# ============================================================================

# Embeddings optimizados para Azure AI Search
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
    api_key=settings.AZURE_OPENAI_API_KEY,
    chunk_size=16,
)

search_service = AzureAISearchService()

# Herramienta de búsqueda técnica
search_tool = StructuredTool.from_function(
    coroutine=search_service.search_technical_docs,
    name="search_technical_docs",
    description="Busca información técnica oficial sobre servicios de Microsoft Azure.",
)

tools_map = {search_tool.name: search_tool}
llm_with_tools = llm.bind_tools([search_tool])

# ============================================================================
# 2. NODOS DEL GRAFO
# ============================================================================


async def agent_node(state: GraphState) -> dict:
    """
    Nodo Agente: Decide si responder o usar herramientas.
    """
    # 1. Clasificación de intención (Solo si no ha sido clasificada en este turno)
    intent = state.get("intention")
    if not intent:
        intent_response = await classify_intent(state["input"])
        intent = intent_response.intention

    if intent == "FUERA_DE_DOMINIO":
        return {"intention": "FUERA_DE_DOMINIO"}

    # 2. Preparación de mensajes
    # LangGraph con MemorySaver inyecta el historial automáticamente en 'history'
    messages = state.get("history", [])

    # Si es el primer mensaje, inyectamos el SystemMessage
    if not messages:
        messages = [
            SystemMessage(
                content=(
                    "Eres un experto en Azure. Pasos:\n"
                    "1. Usa siempre 'search_technical_docs' para datos técnicos.\n"
                    "2. Cita fuentes explícitamente.\n"
                    "3. Si no sabes, admítelo."
                )
            )
        ]

    # Añadimos la consulta actual si no está ya en el historial
    # Nota: El checkpointer maneja la persistencia, nosotros solo preparamos el envío al LLM
    current_input = HumanMessage(content=state["input"])
    full_context = messages + [current_input]

    # 3. Invocación al LLM
    # Importante: El streaming ocurre aquí internamente si llm tiene streaming=True
    response = await llm_with_tools.ainvoke(full_context)

    # Retornamos el input para mantenerlo en el estado si es necesario,
    # y el nuevo mensaje para que el checkpointer lo guarde.
    return {
        "history": [current_input, response],
        "response": response.content,
        "intention": intent,
    }


async def manual_tool_node(state: GraphState) -> dict:
    """
    Ejecutor de herramientas manual para mayor control sobre las fuentes.
    """
    last_message = state["history"][-1]
    tool_outputs = []
    new_sources = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools_map:
            # Ejecución asíncrona de Azure AI Search
            observation = await tools_map[tool_name].ainvoke(tool_call["args"])

            content = observation.get("content", "")
            sources = observation.get("sources", [])

            tool_outputs.append(
                ToolMessage(
                    content=content, tool_call_id=tool_call["id"], name=tool_name
                )
            )
            new_sources.extend(sources)

    return {
        "history": tool_outputs,
        "sources": list(set(state.get("sources", []) + new_sources)),
    }


async def out_of_domain_node(state: GraphState) -> dict:
    """Manejo de preguntas no relacionadas con Azure."""
    response = "Soy un asistente especializado en Azure. ¿En qué servicio de la nube puedo ayudarte?"
    return {"response": response}


# ============================================================================
# 3. LÓGICA DE CONTROL (EDGES)
# ============================================================================


def should_continue(state: GraphState):
    if state.get("intention") == "FUERA_DE_DOMINIO":
        return "out_of_domain"

    last_message = state["history"][-1]
    if last_message.tool_calls:
        return "tools"

    return END


# ============================================================================
# 4. CONSTRUCCIÓN Y COMPILACIÓN
# ============================================================================

workflow = StateGraph(GraphState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", manual_tool_node)
workflow.add_node("out_of_domain", out_of_domain_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "out_of_domain": "out_of_domain", END: END},
)

workflow.add_edge("tools", "agent")
workflow.add_edge("out_of_domain", END)

# Checkpointer para memoria persistente (en memoria para este ejemplo)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
