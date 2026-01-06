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
    current_input = state["input"]
    if not history:
        history = [HumanMessage(content=current_input)]
    else:
        if history[-1].content != current_input:
            history.append(HumanMessage(content=current_input))

    # PROMPT DE MEMORIA SELECTIVA
    sys_msg = SystemMessage(
        content=(
            "Eres un Arquitecto de Azure. Tienes acceso al historial para entender el contexto global, "
            "Responde en UN SOLO PÁRRAFO de máximo 12 líneas."
            "pero para esta respuesta debes priorizar EXCLUSIVAMENTE los nuevos datos de las herramientas. "
            "Si el usuario cambia de tema (ej. de Redes a SQL), olvida lo anterior y enfócate en los nuevos documentos. "
            "Responde en UN SOLO PÁRRAFO (máx 12 líneas) citando Archivo y Página."
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
    """Filtra fuentes de la interacción actual sin romper la cadena del historial."""
    history = state.get("history", [])
    current_response_sources = []
    seen_keys = set()

    # Buscamos fuentes de la ÚLTIMA búsqueda realizada en este turno
    for msg in reversed(history):
        if isinstance(msg, ToolMessage):
            try:
                data = (
                    json.loads(msg.content)
                    if isinstance(msg.content, str)
                    else msg.content
                )
                docs = data if isinstance(data, list) else data.get("value", [])

                for doc in docs:
                    source_name = doc.get("source") or doc.get("title") or "Documento"
                    page = doc.get("page_number") or doc.get("page")

                    clean_name = (
                        str(source_name)
                        .split("\\")[-1]
                        .split("/")[-1]
                        .replace(".pdf", "")
                        .replace(".docx", "")
                    )
                    source_key = f"{clean_name}-{page}"

                    if source_key not in seen_keys:
                        seen_keys.add(source_key)
                        current_response_sources.append(
                            {
                                "title": f"{clean_name}.pdf",
                                "search_term": clean_name.lower(),
                                "page": page,
                                "url": doc.get("url") or "#",
                            }
                        )
            except Exception as e:
                logger.error(f"Error parseando fuentes: {e}")
        if isinstance(msg, HumanMessage):
            break

    # Obtenemos el último texto del asistente
    last_ai_msg = next(
        (m for m in reversed(history) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    response_text = last_ai_msg.content if last_ai_msg else ""

    final_sources = [
        s
        for s in current_response_sources
        if s["search_term"] in response_text.lower()
        or f"página {s['page']}" in response_text.lower()
    ]

    return {
        "response": response_text,
        "sources": final_sources,
        "intention": state.get("intention", "AZURE_ARCHITECT"),
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
