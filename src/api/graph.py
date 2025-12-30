import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.api.core.config import settings
from src.api.core.state import GraphState
from src.api.llm.classifier import classify_intent
from src.api.llm.client import llm
from src.api.search.service import AzureAISearchService

logger = logging.getLogger(__name__)

# --- Componentes ---
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
    api_key=settings.AZURE_OPENAI_API_KEY,
    chunk_size=16,
)

search_service = AzureAISearchService()
search_tool = StructuredTool.from_function(
    coroutine=search_service.search_technical_docs,
    name="search_technical_docs",
    description="BUSCAR SIEMPRE AQUÍ. Proporciona documentos técnicos sobre Azure.",
)

tools_map = {search_tool.name: search_tool}
llm_with_tools = llm.bind_tools([search_tool])

# --- Nodos ---


async def agent_node(state: GraphState) -> dict:
    """Nodo Agente: Fuerza el uso de herramientas y garantiza la aparición de fuentes."""

    intent = state.get("intention")
    if not intent:
        intent_response = await classify_intent(state["input"])
        intent = intent_response.intention

    if intent == "FUERA_DE_DOMINIO":
        return {"intention": "FUERA_DE_DOMINIO", "response": "Tema fuera de Azure."}

    messages = state.get("history", [])

    if not messages:
        messages = [
            SystemMessage(
                content=(
                    "Eres un experto en Azure. OBLIGATORIO:\n"
                    "1. Usa 'search_technical_docs' para cualquier duda técnica.\n"
                    "2. Responde basándote SOLO en los datos recibidos de la herramienta.\n"
                    "3. Al final de tu respuesta, DEBES escribir: '### 📚 Fuentes consultadas:' seguido de los nombres de los archivos."
                )
            )
        ]
        messages.append(HumanMessage(content=state["input"]))

    # Invocación al LLM
    response = await llm_with_tools.ainvoke(messages)

    # --- LÓGICA DE GARANTÍA DE FUENTES ---
    final_content = response.content
    sources = state.get("sources", [])

    # Si el modelo ya terminó de responder (no hay tool_calls) pero no puso las fuentes:
    if (
        not response.tool_calls
        and sources
        and "Fuentes consultadas" not in final_content
    ):
        sources_list = "\n".join([f"- {s}" for s in set(sources)])
        final_content += f"\n\n### 📚 Fuentes consultadas:\n{sources_list}"

    return {"history": [response], "response": final_content, "intention": intent}


async def manual_tool_node(state: GraphState) -> dict:
    """Ejecutor de herramientas: Almacena los nombres de archivos en el estado."""
    last_message = state["history"][-1]
    tool_outputs = []
    new_sources = []

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            logger.info(f"🛠️ Consultando Azure AI Search para: {tool_call['args']}")
            observation = await tools_map[tool_call["name"]].ainvoke(tool_call["args"])

            tool_outputs.append(
                ToolMessage(
                    content=observation.get("content", ""),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
            # Extraemos los nombres de los archivos (sources)
            if observation.get("sources"):
                new_sources.extend(observation["sources"])

    return {
        "history": tool_outputs,
        "sources": list(set(state.get("sources", []) + new_sources)),
    }


async def out_of_domain_node(state: GraphState) -> dict:
    return {"response": "Solo puedo ayudarte con temas técnicos de Microsoft Azure."}


# --- Control de flujo ---


def should_continue(state: GraphState):
    if state.get("intention") == "FUERA_DE_DOMINIO":
        return "out_of_domain"

    messages = state.get("history", [])
    if not messages:
        return END

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


# --- Construcción del Grafo ---

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

app = workflow.compile(checkpointer=MemorySaver())
