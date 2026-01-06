import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Asegúrate de que en tu GraphState existan estos campos:
# class GraphState(TypedDict):
#     input: str
#     intention: str
#     history: List[BaseMessage]
#     response: str
#     sources: List[dict]
from src.api.core.state import GraphState
from src.api.llm.classifier import classify_intent
from src.api.llm.client import llm
from src.api.search.service import AzureAISearchService

logger = logging.getLogger(__name__)

search_service = AzureAISearchService()
search_tool = StructuredTool.from_function(
    coroutine=search_service.search_technical_docs,
    name="search_technical_docs",
    description="Proporciona documentos técnicos oficiales sobre Azure.",
)
tools = [search_tool]
tool_node = ToolNode(tools)

# --- NODOS ---


async def router_node(state: GraphState) -> dict:
    intent_response = await classify_intent(state["input"])
    return {"intention": intent_response.intention}


async def agent_node(state: GraphState) -> dict:
    # Inicializar historial si está vacío
    history = state.get("history", [])
    if not history:
        history = [HumanMessage(content=state["input"])]

    sys_msg = SystemMessage(
        content=(
            "Eres Arquitecto Senior de Azure. Responde en UN SOLO PÁRRAFO de máximo 20 líneas. "
            "DEBES usar la herramienta search_technical_docs para fundamentar tu respuesta técnica. "
            "Cita las fuentes dentro del texto (ej. Manual_Redes_v1.pdf, pág. 5)."
        )
    )

    # Vinculamos herramientas y llamamos al modelo
    model_with_tools = llm.bind_tools(tools)
    response = await model_with_tools.ainvoke([sys_msg] + history)

    # Añadimos la respuesta al historial para que el grafo siga el rastro
    return {"history": history + [response]}


async def custom_tool_wrapper(state: GraphState) -> dict:
    # Ejecuta la herramienta basada en el último mensaje de tipo 'tool_calls'
    result = await tool_node.ainvoke({"messages": state["history"]})
    # Retornamos el historial acumulado
    return {"history": state["history"] + result["messages"]}


async def finalize_node(state: GraphState) -> dict:
    """Extrae la respuesta final y las fuentes de los ToolMessages."""
    history = state.get("history", [])

    # 1. Obtener el texto de la última respuesta del asistente (la que ya tiene la info de la herramienta)
    last_ai_msg = next(
        (
            msg
            for msg in reversed(history)
            if isinstance(msg, AIMessage) and not msg.tool_calls
        ),
        None,
    )
    response_text = (
        last_ai_msg.content
        if last_ai_msg
        else "Lo siento, no pude generar una respuesta técnica."
    )

    found_sources = []
    seen_keys = set()

    # 2. Extraer metadatos de los mensajes de herramienta
    for msg in history:
        if isinstance(msg, ToolMessage):
            try:
                # El contenido de Azure AI Search suele venir como JSON string o lista
                content = msg.content
                data = json.loads(content) if isinstance(content, str) else content

                # Normalizar si viene en el campo 'value' (estándar de Azure SDK)
                docs = data if isinstance(data, list) else data.get("value", [])

                for doc in docs:
                    source_name = doc.get("source")  # Campo de tu JSON
                    page = doc.get("page_number")  # Campo de tu JSON

                    if source_name:
                        # Limpiar ruta si fuera necesario
                        clean_name = str(source_name).split("\\")[-1].split("/")[-1]
                        source_key = f"{clean_name}-{page}"

                        if source_key not in seen_keys:
                            seen_keys.add(source_key)
                            found_sources.append(
                                {
                                    "title": clean_name,
                                    "page": page,
                                    "url": doc.get("url")
                                    or doc.get("metadata_storage_path", "#"),
                                }
                            )
            except Exception as e:
                logger.error(f"Error parseando fuentes en finalize_node: {e}")

    return {
        "response": response_text,
        "sources": found_sources,
        "history": history,  # Mantenemos el historial actualizado
    }


async def out_of_domain_node(state: GraphState) -> dict:
    msg = "Solo puedo ayudarte con temas técnicos de Microsoft Azure."
    return {
        "response": msg,
        "sources": [],
        "history": state["history"] + [AIMessage(content=msg)],
    }


# --- LÓGICA DE CONTROL ---


def should_continue(state: GraphState):
    history = state.get("history", [])
    last_message = history[-1]

    # Si el último mensaje tiene llamadas a herramientas, vamos a 'tools'
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Si no, el agente ya respondió basado en la info de la herramienta
    return "finalize"


def route_after_classifier(state: GraphState):
    if state.get("intention") == "FUERA_DE_DOMINIO":
        return "out_of_domain"
    return "agent"


# --- CONSTRUCCIÓN DEL FLUJO ---

workflow = StateGraph(GraphState)

workflow.add_node("router", router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", custom_tool_wrapper)
workflow.add_node("finalize", finalize_node)
workflow.add_node("out_of_domain", out_of_domain_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", route_after_classifier)

workflow.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "finalize": "finalize"}
)

# El ciclo: después de usar la herramienta, vuelve al agente para que redacte
workflow.add_edge("tools", "agent")

workflow.add_edge("finalize", END)
workflow.add_edge("out_of_domain", END)

app = workflow.compile(checkpointer=MemorySaver())
