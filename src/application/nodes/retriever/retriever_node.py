import json
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from src.adapters.azure.ai_search import AzureAISearchService
from src.domain.entities.schemas import SearchTechnicalDocsInput

logger = logging.getLogger(__name__)

search_service = AzureAISearchService()
search_tool = StructuredTool.from_function(
    coroutine=search_service.search_technical_docs,
    name="search_technical_docs",
    description=(
        "Busca en documentos técnicos oficiales de Azure (PDFs de redes, seguridad, SQL). "
        "IMPORTANTE: Usa SIEMPRE la pregunta ACTUAL del usuario como parámetro de búsqueda, "
        "no uses preguntas anteriores del historial."
    ),
    args_schema=SearchTechnicalDocsInput,
)
tools = [search_tool]
tool_node = ToolNode(tools)

logger.info(f"Herramientas configuradas: {[t.name for t in tools]}")
logger.info(
    f"Herramienta 'search_technical_docs' tiene esquema: {search_tool.args_schema}"
)


async def retriever_node(state: dict) -> dict:
    """
    Wrapper robusto para ejecutar herramientas del agente.
    """
    try:
        history = state.get("history", [])

        last_message = history[-1] if history else None
        if not last_message or not isinstance(last_message, AIMessage):
            logger.warning(
                "retriever_node - Último mensaje no es AIMessage, saltando ejecución"
            )
            return {"history": history}

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.warning(
                "retriever_node - Último mensaje no tiene tool_calls, saltando ejecución"
            )
            return {"history": history}

        logger.info(
            f"retriever_node - Ejecutando herramientas con historial de {len(history)} mensajes"
        )
        logger.info(
            f"retriever_node - Tool calls detectados: {len(last_message.tool_calls)}"
        )

        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_call_id = tool_call.get("id", "")
            tool_args = tool_call.get("args", {})

            logger.info(
                f"retriever_node - Procesando tool_call: name={tool_name}, id={tool_call_id}"
            )
            logger.info(f"retriever_node - Args recibidos: {tool_args}")

            try:
                if not tool_name:
                    raise ValueError("Tool name está vacío")

                if not tool_call_id:
                    raise ValueError("Tool call ID está vacío")

                tool = next((t for t in tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(
                        f"Herramienta '{tool_name}' no encontrada. "
                        f"Herramientas disponibles: {[t.name for t in tools]}"
                    )

                if tool_name == "search_technical_docs":
                    if isinstance(tool_args, dict):
                        query = tool_args.get("query", "")
                        if not query or not isinstance(query, str):
                            query = str(
                                tool_args.get(
                                    "query", tool_args.get("text", tool_args.get("input", ""))
                                )
                            )
                            if not query:
                                raise ValueError(
                                    f"Argumento 'query' no encontrado en tool_args: {tool_args}"
                                )
                        tool_args = {"query": query}
                    elif isinstance(tool_args, str):
                        tool_args = {"query": tool_args}
                    else:
                        raise ValueError(f"Argumentos inválidos para {tool_name}: {tool_args}")

                logger.info(
                    f"retriever_node - Ejecutando herramienta '{tool_name}' con args normalizados: {tool_args}"
                )

                try:
                    if hasattr(tool, "ainvoke"):
                        result = await tool.ainvoke(tool_args)
                    elif hasattr(tool, "invoke"):
                        result = tool.invoke(tool_args)
                    else:
                        raise ValueError(
                            f"Herramienta '{tool_name}' no tiene métodos ainvoke o invoke"
                        )

                    logger.info(
                        f"retriever_node - Herramienta '{tool_name}' ejecutada exitosamente"
                    )

                except Exception as exec_error:
                    logger.error(
                        f"retriever_node - Error durante ejecución de '{tool_name}': {str(exec_error)}",
                        exc_info=True,
                    )
                    error_content = json.dumps(
                        {
                            "error": "Error ejecutando herramienta",
                            "message": str(exec_error)[:500],
                            "tool_name": tool_name,
                            "error_type": type(exec_error).__name__,
                            "content": (
                                f"No se pudo ejecutar la herramienta '{tool_name}'. "
                                "Por favor, intenta reformular tu pregunta."
                            ),
                            "value": [],
                        },
                        ensure_ascii=False,
                    )

                    error_msg = ToolMessage(
                        content=error_content,
                        tool_call_id=tool_call_id if tool_call_id else f"error_{len(tool_messages)}",
                    )
                    tool_messages.append(error_msg)
                    logger.warning(
                        f"retriever_node - ToolMessage de error creado para '{tool_name}' "
                        "(en lugar de re-lanzar)"
                    )
                    continue

                if isinstance(result, dict):
                    try:
                        content = json.dumps(result, ensure_ascii=False, default=str)
                    except (TypeError, ValueError) as json_error:
                        logger.warning(
                            f"retriever_node - Error serializando dict, usando str: {json_error}"
                        )
                        content = json.dumps(
                            {"content": str(result), "value": []}, ensure_ascii=False
                        )
                elif isinstance(result, str):
                    content = json.dumps({"content": result, "value": []}, ensure_ascii=False)
                else:
                    content = json.dumps({"content": str(result), "value": []}, ensure_ascii=False)

                tool_msg = ToolMessage(content=content, tool_call_id=tool_call_id)
                tool_messages.append(tool_msg)
                logger.info(
                    f"retriever_node - ToolMessage creado para '{tool_name}' (content_length={len(content)})"
                )

            except Exception as tool_error:
                logger.error(
                    f"retriever_node - Error procesando tool_call '{tool_name}': {str(tool_error)}",
                    exc_info=True,
                )
                error_content = json.dumps(
                    {
                        "error": "Error ejecutando herramienta",
                        "message": str(tool_error)[:500],
                        "tool_name": tool_name,
                        "error_type": type(tool_error).__name__,
                        "content": (
                            f"No se pudo ejecutar la herramienta '{tool_name}'. "
                            "Por favor, intenta reformular tu pregunta."
                        ),
                        "value": [],
                    },
                    ensure_ascii=False,
                )

                error_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call_id if tool_call_id else f"error_{len(tool_messages)}",
                )
                tool_messages.append(error_msg)
                logger.warning(
                    f"retriever_node - ToolMessage de error creado para '{tool_name}'"
                )

        tool_call_ids = {tc.get("id") for tc in last_message.tool_calls if tc.get("id")}
        tool_message_ids = {
            msg.tool_call_id
            for msg in tool_messages
            if isinstance(msg, ToolMessage)
            and hasattr(msg, "tool_call_id")
            and msg.tool_call_id
        }

        missing_ids = tool_call_ids - tool_message_ids
        if missing_ids:
            logger.error(f"retriever_node - CRÍTICO: Tool calls sin respuesta: {missing_ids}")
            for tool_call in last_message.tool_calls:
                if tool_call.get("id") in missing_ids:
                    error_msg = ToolMessage(
                        content=json.dumps(
                            {
                                "error": "Error ejecutando herramienta",
                                "message": "No se pudo procesar esta herramienta",
                                "content": (
                                    "No se pudo ejecutar la herramienta. Por favor, intenta reformular tu pregunta."
                                ),
                                "value": [],
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=tool_call.get("id", f"missing_{len(tool_messages)}"),
                    )
                    tool_messages.append(error_msg)

        return {"history": history + tool_messages}

    except Exception as wrapper_error:
        logger.critical(
            f"retriever_node - ERROR CRÍTICO en el wrapper: {str(wrapper_error)}",
            exc_info=True,
        )

        history = state.get("history", [])
        last_message = history[-1] if history else None

        if last_message and isinstance(last_message, AIMessage):
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                error_messages = []
                for tool_call in last_message.tool_calls:
                    tool_call_id = tool_call.get("id", f"error_{len(error_messages)}")
                    error_msg = ToolMessage(
                        content=json.dumps(
                            {
                                "error": "Error crítico en el sistema",
                                "message": "No se pudo procesar las herramientas correctamente",
                                "content": (
                                    "Hubo un error al procesar tu solicitud. Por favor, intenta nuevamente."
                                ),
                                "value": [],
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=tool_call_id,
                    )
                    error_messages.append(error_msg)

                logger.warning(
                    "retriever_node - Retornando "
                    f"{len(error_messages)} ToolMessages de error por fallo crítico"
                )
                return {"history": history + error_messages}

        logger.warning("retriever_node - No hay tool_calls, devolviendo historial sin cambios")
        return {"history": history}
