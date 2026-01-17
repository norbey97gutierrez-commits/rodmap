import json
import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.adapters.azure.openai_client import llm
from src.application.nodes.classifier.classifier_node import classify_intent
from src.application.nodes.extractor.extractor_node import extractor_node
from src.application.nodes.retriever.retriever_node import retriever_node, tools
from src.application.state import GraphState

logger = logging.getLogger(__name__)


def _get_checkpointer():
    return MemorySaver()


async def router_node(state: GraphState) -> dict:
    current_input = state.get("input", "")
    existing_history = state.get("history", [])

    logger.info(f"router_node - Clasificando input: '{current_input[:100]}...'")
    logger.info(
        f"router_node - Historial existente tiene {len(existing_history)} mensajes"
    )

    new_history = [HumanMessage(content=current_input)]

    last_human = next(
        (m for m in reversed(existing_history) if isinstance(m, HumanMessage)), None
    )
    if last_human:
        if last_human.content.strip() != current_input.strip():
            logger.info(
                "router_node - Input diferente detectado (dejar que merge_history decida)"
            )
        else:
            logger.info("router_node - Mismo input (continuación de conversación)")
    else:
        logger.info("router_node - Primera consulta")

    intent_response = await classify_intent(current_input)
    logger.info(f"router_node - Intención clasificada: {intent_response.intention}")

    return {
        "intention": intent_response.intention,
        "input": current_input,
        "history": new_history,
    }


def _validate_and_filter_history(history: list) -> list:
    if not history:
        return []

    tool_messages_by_id = {}
    for msg in history:
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id and tool_call_id not in tool_messages_by_id:
                tool_messages_by_id[tool_call_id] = msg

    validated_history = []
    i = 0
    while i < len(history):
        msg = history[i]

        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_ids = [tc.get("id") for tc in msg.tool_calls if tc.get("id")]
            validated_history.append(msg)

            for tool_call_id in tool_call_ids:
                if tool_call_id in tool_messages_by_id:
                    original_tool_msg = tool_messages_by_id[tool_call_id]
                    tool_msg = ToolMessage(
                        content=original_tool_msg.content,
                        tool_call_id=original_tool_msg.tool_call_id,
                    )
                    validated_history.append(tool_msg)
                else:
                    error_tool_msg = ToolMessage(
                        content=json.dumps(
                            {
                                "error": "ToolMessage faltante",
                                "message": "No se encontró el ToolMessage correspondiente",
                                "content": "Error al recuperar información de la herramienta.",
                                "value": [],
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=tool_call_id,
                    )
                    validated_history.append(error_tool_msg)

            i += 1
            while i < len(history) and isinstance(history[i], ToolMessage):
                i += 1
        else:
            if not isinstance(msg, ToolMessage):
                validated_history.append(msg)
            i += 1

    return validated_history


async def agent_node(state: GraphState) -> dict:
    history = state.get("history", [])
    current_input = state.get("input", "")

    if not any(isinstance(m, HumanMessage) for m in history):
        history = [HumanMessage(content=current_input)] + history

    validated_history = _validate_and_filter_history(history)

    sys_msg = SystemMessage(
        content=(
            "Eres un Arquitecto de Azure experto. Tu objetivo es responder ÚNICAMENTE a la ÚLTIMA pregunta del usuario.\n\n"
            f"PREGUNTA ACTUAL DEL USUARIO: '{current_input}'\n\n"
            "INSTRUCCIONES CRÍTICAS:\n"
            "1. Responde SOLO a la pregunta actual mencionada arriba.\n"
            "2. Responde en UN SOLO PÁRRAFO de máximo 12 líneas.\n"
            "3. Usa ÚNICAMENTE los documentos recuperados que sean RELEVANTES a la pregunta ACTUAL.\n"
            "4. Si los documentos hablan de SQL pero la pregunta es de Redes, IGNORA los de SQL.\n"
            "5. Cita SIEMPRE el nombre del archivo que encuentres en los documentos.\n"
            "6. NO mezcles información de temas distintos.\n"
            "7. Cuando uses la herramienta de búsqueda, usa SIEMPRE la pregunta ACTUAL del usuario.\n"
            "8. NO repitas respuestas anteriores. Genera una respuesta NUEVA y ÚNICA basada en la pregunta actual.\n"
            "9. Si la pregunta cambió, ignora completamente las respuestas anteriores y genera una nueva."
        )
    )

    try:
        if not llm:
            raise ValueError("LLM no está inicializado correctamente")

        if not tools:
            response = await llm.ainvoke([sys_msg] + validated_history)
        else:
            model_with_tools = llm.bind_tools(tools)
            response = await model_with_tools.ainvoke([sys_msg] + validated_history)

        if not response:
            raise ValueError("El LLM no generó una respuesta válida")

        return {
            "history": history + [response],
            "input": current_input,
        }
    except Exception as e:
        error_message = str(e)
        if "tool" in error_message.lower() or "tool_calls" in error_message.lower():
            error_content = (
                "Lo siento, hubo un problema con la ejecución de herramientas de la IA. "
                "Por favor, intenta reformular tu pregunta."
            )
        elif "timeout" in error_message.lower():
            error_content = (
                "La solicitud tardó demasiado tiempo. Por favor, intenta con una pregunta más específica."
            )
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            error_content = (
                "Error de conexión con los servicios de Azure. "
                "Por favor, intenta nuevamente en unos momentos."
            )
        else:
            error_content = (
                "Lo siento, hubo un error al procesar tu solicitud. Por favor, intenta reformular tu pregunta."
            )

        error_msg = AIMessage(content=error_content)
        if hasattr(error_msg, "tool_calls"):
            error_msg.tool_calls = None

        return {"history": history + [error_msg], "input": current_input}


async def out_of_domain_node(state: GraphState) -> dict:
    msg = "Solo puedo ayudarte con temas técnicos de Microsoft Azure."
    return {
        "response": msg,
        "sources": [],
        "history": state.get("history", []) + [AIMessage(content=msg)],
    }


def should_continue(state: GraphState):
    history = state.get("history", [])
    if not history:
        return "finalize"
    last_message = history[-1]
    if (
        hasattr(last_message, "tool_calls")
        and last_message.tool_calls
        and len(last_message.tool_calls) > 0
    ):
        return "retriever"
    return "finalize"


def route_after_classifier(state: GraphState):
    if state.get("intention") == "FUERA_DE_DOMINIO":
        return "out_of_domain"
    return "agent"


workflow = StateGraph(GraphState)

workflow.add_node("router", router_node)
workflow.add_node("agent", agent_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("finalize", extractor_node)
workflow.add_node("out_of_domain", out_of_domain_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_after_classifier)
workflow.add_conditional_edges(
    "agent", should_continue, {"retriever": "retriever", "finalize": "finalize"}
)
workflow.add_edge("retriever", "agent")
workflow.add_edge("finalize", END)
workflow.add_edge("out_of_domain", END)

def build_graph(checkpointer=None):
    return workflow.compile(checkpointer=checkpointer or MemorySaver())


app = build_graph(_get_checkpointer())
