import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.api.core.state import GraphState
from src.api.llm.classifier import classify_intent
from src.api.llm.client import llm
from src.api.search.service import AzureAISearchService

logger = logging.getLogger(__name__)

# CONFIGURACIÓN DE HERRAMIENTAS
search_service = AzureAISearchService()
search_tool = StructuredTool.from_function(
    coroutine=search_service.search_technical_docs,
    name="search_technical_docs",
    description="Busca en documentos técnicos oficiales de Azure (PDFs de redes, seguridad, SQL).",
)
tools = [search_tool]
tool_node = ToolNode(tools)

# NODOS DEL GRAFO


async def router_node(state: GraphState) -> dict:
    intent_response = await classify_intent(state["input"])
    return {"intention": intent_response.intention}


async def agent_node(state: GraphState) -> dict:
    """Mantiene la continuidad pero prioriza los datos nuevos."""
    history = state.get("history", [])
    # CAMBIO: Usamos state.get("text") porque es lo que envía el test/frontend
    current_input = state.get("text") or state.get("input")

    if not history:
        history = [HumanMessage(content=current_input)]

    sys_msg = SystemMessage(
        content=(
            "Eres un Arquitecto de Azure. Tu objetivo es responder ÚNICAMENTE a la última pregunta "
            "Responde en UN SOLO PÁRRAFO de máximo 12 líneas."
            "usando los documentos recuperados que sean RELEVANTES a ese tema específico. "
            "Si los documentos recuperados hablan de SQL pero la pregunta es de Redes, IGNORA los de SQL. "
            "Cita SIEMPRE el nombre del archivo que encuentres en el documents.json. "
            "No mezcles información de temas distintos."
        )
    )

    model_with_tools = llm.bind_tools(tools)
    response = await model_with_tools.ainvoke([sys_msg] + history)

    return {"history": history + [response]}


async def custom_tool_wrapper(state: GraphState) -> dict:
    # Ejecutamos la búsqueda basada en el historial
    result = await tool_node.ainvoke({"messages": state["history"]})
    return {"history": state["history"] + result["messages"]}


async def finalize_node(state: GraphState) -> dict:
    history = state.get("history", [])
    last_ai_msg = next(
        (m for m in reversed(history) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    response_text = last_ai_msg.content if last_ai_msg else ""

    current_response_sources = []
    seen_keys = set()

    # Solo buscamos fuentes en el último mensaje de herramientas (el más reciente)
    for msg in reversed(history):
        if isinstance(msg, ToolMessage):
            try:
                data = (
                    json.loads(msg.content)
                    if isinstance(msg.content, str)
                    else msg.content
                )
                docs = data.get("value", []) if isinstance(data, dict) else []

                for doc in docs:
                    source_name = doc.get("source") or doc.get("title") or ""
                    # Limpiamos para tener solo el nombre base: Manual_Redes_v1
                    clean_name = (
                        str(source_name)
                        .split("\\")[-1]
                        .split("/")[-1]
                        .replace(".pdf", "")
                        .replace(".docx", "")
                    )

                    # VALIDACIÓN CRÍTICA: ¿El nombre del archivo está en el texto de la respuesta?
                    # Usamos una búsqueda insensible a mayúsculas
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
            except Exception:
                continue

        # Al llegar al mensaje del humano, dejamos de buscar herramientas de turnos anteriores
        if isinstance(msg, HumanMessage):
            break

    return {
        "response": response_text,
        "sources": current_response_sources,
        "history": history,
    }


async def out_of_domain_node(state: GraphState) -> dict:
    msg = "Solo puedo ayudarte con temas técnicos de Microsoft Azure."
    return {
        "response": msg,
        "sources": [],
        "history": state.get("history", []) + [AIMessage(content=msg)],
    }


# LÓGICA DE CONTROL


def should_continue(state: GraphState):
    history = state.get("history", [])
    if not history:
        return "finalize"
    last_message = history[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "finalize"


def route_after_classifier(state: GraphState):
    if state.get("intention") == "FUERA_DE_DOMINIO":
        return "out_of_domain"
    return "agent"


# CONSTRUCCIÓN DEL GRAPH

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
workflow.add_edge("tools", "agent")
workflow.add_edge("finalize", END)
workflow.add_edge("out_of_domain", END)

app = workflow.compile(checkpointer=MemorySaver())
