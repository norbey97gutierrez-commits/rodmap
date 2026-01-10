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
from pydantic import BaseModel, Field

# Esquema para los argumentos de la herramienta
class SearchTechnicalDocsInput(BaseModel):
    query: str = Field(..., description="La pregunta o consulta del usuario sobre Azure (VNets, SQL, escalabilidad, etc.)")

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
logger.info(f"Herramienta 'search_technical_docs' tiene esquema: {search_tool.args_schema}")

# NODOS DEL GRAFO


async def router_node(state: GraphState) -> dict:
    """
    Nodo de enrutamiento que clasifica la intención y prepara el historial.
    
    ESTRATEGIA ROBUSTA:
    - Siempre pasamos solo el nuevo HumanMessage con el input actual
    - Dejamos que merge_history_with_reset decida si reiniciar o no el historial
    - Esto evita lógica duplicada y conflictos entre router_node y merge_history_with_reset
    """
    current_input = state.get("input", "")
    existing_history = state.get("history", [])
    
    logger.info(f"router_node - Clasificando input: '{current_input[:100]}...'")
    logger.info(f"router_node - Historial existente tiene {len(existing_history)} mensajes")
    
    # Siempre pasamos solo el nuevo HumanMessage
    # merge_history_with_reset decidirá si reiniciar o no basándose en:
    # 1. Si el input cambió respecto al último HumanMessage
    # 2. Si hay tool_calls pendientes que deben completarse primero
    new_history = [HumanMessage(content=current_input)]
    
    # Logging informativo
    last_human = next(
        (m for m in reversed(existing_history) if isinstance(m, HumanMessage)),
        None
    )
    if last_human:
        if last_human.content.strip() != current_input.strip():
            logger.info(f"router_node - Input diferente detectado (dejar que merge_history decida)")
        else:
            logger.info(f"router_node - Mismo input (continuación de conversación)")
    else:
        logger.info(f"router_node - Primera consulta")
    
    # Clasificamos la intención
    intent_response = await classify_intent(current_input)
    logger.info(f"router_node - Intención clasificada: {intent_response.intention}")
    
    return {
        "intention": intent_response.intention,
        "input": current_input,
        "history": new_history  # Siempre solo el nuevo HumanMessage
    }


def _validate_and_filter_history(history: list) -> list:
    """
    Construye un historial válido asegurando que los AIMessages con tool_calls
    estén seguidos inmediatamente por sus ToolMessages correspondientes.
    
    CRÍTICO: Azure OpenAI requiere que cada AIMessage con tool_calls esté seguido
    inmediatamente por sus ToolMessages. Si no, lanza un error 400.
    
    ESTRATEGIA CORREGIDA:
    1. Construimos un nuevo historial desde cero
    2. Para cada AIMessage con tool_calls, lo agregamos seguido de TODOS sus ToolMessages
    3. Los ToolMessages se buscan en TODO el historial y se colocan inmediatamente después
    4. IMPORTANTE: Si un ToolMessage ya fue usado, creamos una COPIA para el segundo AIMessage
    5. Si faltan ToolMessages, creamos ToolMessages de error (nunca omitimos el AIMessage)
    """
    if not history:
        return []
    
    # Mapeo de todos los ToolMessages por su tool_call_id
    tool_messages_by_id = {}
    for msg in history:
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, 'tool_call_id', None)
            if tool_call_id:
                # Guardamos el primer ToolMessage encontrado para cada ID
                if tool_call_id not in tool_messages_by_id:
                    tool_messages_by_id[tool_call_id] = msg
    
    logger.info(f"_validate_and_filter_history - Total ToolMessages únicos en historial: {len(tool_messages_by_id)}")
    logger.info(f"_validate_and_filter_history - Tool call IDs disponibles: {list(tool_messages_by_id.keys())}")
    
    validated_history = []
    
    i = 0
    while i < len(history):
        msg = history[i]
        
        # Si es un AIMessage con tool_calls
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_call_ids = [tc.get('id') for tc in msg.tool_calls if tc.get('id')]
            logger.info(f"_validate_and_filter_history - AIMessage en posición {i} con {len(tool_call_ids)} tool_calls: {tool_call_ids}")
            
            # Agregamos el AIMessage
            validated_history.append(msg)
            
            # Buscamos y agregamos TODOS sus ToolMessages inmediatamente después
            for tool_call_id in tool_call_ids:
                if tool_call_id in tool_messages_by_id:
                    # IMPORTANTE: Creamos una COPIA del ToolMessage para cada AIMessage
                    # Esto permite que múltiples AIMessages con el mismo tool_call_id tengan sus ToolMessages
                    original_tool_msg = tool_messages_by_id[tool_call_id]
                    # Creamos una copia del ToolMessage
                    tool_msg = ToolMessage(
                        content=original_tool_msg.content,
                        tool_call_id=original_tool_msg.tool_call_id
                    )
                    validated_history.append(tool_msg)
                    logger.info(f"_validate_and_filter_history - ✓ ToolMessage agregado (copia) para {tool_call_id}")
                else:
                    # CRÍTICO: Si falta el ToolMessage, creamos uno de error
                    logger.error(f"_validate_and_filter_history - ✗ ToolMessage faltante para {tool_call_id}, creando error")
                    error_tool_msg = ToolMessage(
                        content=json.dumps({
                            "error": "ToolMessage faltante",
                            "message": "No se encontró el ToolMessage correspondiente",
                            "content": "Error al recuperar información de la herramienta.",
                            "value": []
                        }, ensure_ascii=False),
                        tool_call_id=tool_call_id
                    )
                    validated_history.append(error_tool_msg)
            
            i += 1
            # Saltamos ToolMessages que están inmediatamente después (ya los agregamos arriba)
            while i < len(history) and isinstance(history[i], ToolMessage):
                i += 1
        else:
            # Mensaje normal: HumanMessage, SystemMessage, AIMessage sin tool_calls
            # Solo agregamos si NO es un ToolMessage (los ToolMessages ya se agregaron con sus AIMessages)
            if not isinstance(msg, ToolMessage):
                validated_history.append(msg)
            i += 1
    
    logger.info(f"_validate_and_filter_history - Historial validado: {len(history)} -> {len(validated_history)} mensajes")
    return validated_history


async def agent_node(state: GraphState) -> dict:
    """
    Nodo del agente que genera respuestas usando el LLM.
    
    ESTRATEGIA ROBUSTA:
    - Usa siempre el input actual del estado (no del historial)
    - Valida y filtra el historial antes de enviarlo al LLM
    - Asegura que los AIMessages con tool_calls estén seguidos por sus ToolMessages
    - El prompt del sistema enfatiza responder solo a la pregunta actual
    """
    history = state.get("history", [])
    current_input = state.get("input", "")
    
    logger.info(f"agent_node - Input actual: '{current_input[:100]}...'")
    logger.info(f"agent_node - Historial tiene {len(history)} mensajes")
    
    # Validación: verificamos que el historial tenga al menos un HumanMessage
    if not any(isinstance(m, HumanMessage) for m in history):
        logger.warning(f"agent_node - No hay HumanMessage en el historial, agregando input actual")
        history = [HumanMessage(content=current_input)] + history
    
    # CRÍTICO: Construimos un historial limpio desde el último HumanMessage hacia adelante
    # Esto asegura que cada AIMessage con tool_calls tenga sus ToolMessages inmediatamente después
    validated_history = _validate_and_filter_history(history)
    
    logger.info(f"agent_node - Historial validado: {len(history)} -> {len(validated_history)} mensajes")
    
    # Verificación final: cada AIMessage con tool_calls debe tener sus ToolMessages después
    for idx, msg in enumerate(validated_history):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_call_ids = {tc.get('id') for tc in msg.tool_calls if tc.get('id')}
            # Verificamos que los siguientes mensajes sean ToolMessages con los IDs correctos
            following_tool_ids = set()
            for j in range(idx + 1, min(idx + 1 + len(tool_call_ids), len(validated_history))):
                if isinstance(validated_history[j], ToolMessage):
                    tool_call_id = getattr(validated_history[j], 'tool_call_id', None)
                    if tool_call_id:
                        following_tool_ids.add(tool_call_id)
            
            missing = tool_call_ids - following_tool_ids
            if missing:
                logger.error(f"agent_node - ERROR: AIMessage en posición {idx} tiene tool_calls sin ToolMessages inmediatamente después")
                logger.error(f"agent_node - Tool call IDs esperados: {tool_call_ids}")
                logger.error(f"agent_node - Tool call IDs encontrados: {following_tool_ids}")
                logger.error(f"agent_node - Faltan: {missing}")
            else:
                logger.info(f"agent_node - ✓ AIMessage en posición {idx} tiene todos sus ToolMessages")
    
    # Prompt del sistema que enfatiza responder solo a la pregunta actual
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
        # Validación adicional: verificamos que el LLM esté inicializado
        if not llm:
            raise ValueError("LLM no está inicializado correctamente")
        
        # Validación adicional: verificamos que las herramientas estén disponibles
        if not tools:
            logger.warning(f"agent_node - No hay herramientas disponibles, usando LLM sin herramientas")
            response = await llm.ainvoke([sys_msg] + validated_history)
        else:
            logger.info(f"agent_node - Invocando LLM con {len(tools)} herramienta(s)")
            logger.info(f"agent_node - Historial a enviar: {len(validated_history)} mensajes")
            try:
                model_with_tools = llm.bind_tools(tools)
                logger.info(f"agent_node - LLM con herramientas configurado, invocando...")
                response = await model_with_tools.ainvoke([sys_msg] + validated_history)
                logger.info(f"agent_node - LLM invocado exitosamente")
            except Exception as bind_error:
                logger.error(f"agent_node - Error al invocar LLM con herramientas: {type(bind_error).__name__}: {str(bind_error)}", exc_info=True)
                raise bind_error
        
        logger.info(f"agent_node - Respuesta generada, tiene tool_calls: {bool(hasattr(response, 'tool_calls') and response.tool_calls)}")
        
        # Validación adicional: verificamos que la respuesta sea válida
        if not response:
            raise ValueError("El LLM no generó una respuesta válida")
        
        return {
            "history": history + [response],
            "input": current_input  # Mantenemos el input actualizado en el estado
        }
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        
        logger.error(f"=" * 80)
        logger.error(f"agent_node - ERROR DETALLADO")
        logger.error(f"Tipo: {error_type}")
        logger.error(f"Mensaje: {error_message}")
        logger.error(f"Traceback completo:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"=" * 80)
        
        # Si el error está relacionado con herramientas, lo indicamos en el mensaje
        if "tool" in error_message.lower() or "tool_calls" in error_message.lower():
            error_content = "Lo siento, hubo un problema con la ejecución de herramientas de la IA. Por favor, intenta reformular tu pregunta."
        elif "timeout" in error_message.lower():
            error_content = "La solicitud tardó demasiado tiempo. Por favor, intenta con una pregunta más específica."
        elif "connection" in error_message.lower() or "network" in error_message.lower():
            error_content = "Error de conexión con los servicios de Azure. Por favor, intenta nuevamente en unos momentos."
        else:
            error_content = "Lo siento, hubo un error al procesar tu solicitud. Por favor, intenta reformular tu pregunta."
        
        # Si hay un error, devolvemos el historial con un mensaje de error
        # IMPORTANTE: El mensaje de error debe tener content pero NO tool_calls
        # para que should_continue lo dirija a "finalize" en lugar de "tools"
        error_msg = AIMessage(content=error_content)
        # Aseguramos que no tenga tool_calls
        if hasattr(error_msg, 'tool_calls'):
            error_msg.tool_calls = None
        
        logger.info(f"agent_node - Devolviendo mensaje de error: '{error_content[:100]}...'")
        return {
            "history": history + [error_msg],
            "input": current_input
        }


async def custom_tool_wrapper(state: GraphState) -> dict:
    """
    Wrapper robusto para ejecutar herramientas del agente.
    
    ESTRATEGIA ULTRA-ROBUSTA:
    1. Valida que haya tool_calls válidos
    2. Ejecuta cada herramienta individualmente con manejo de errores completo
    3. Garantiza que TODOS los tool_calls tengan su ToolMessage correspondiente
    4. Nunca falla - siempre devuelve ToolMessages válidos
    """
    try:
        history = state.get("history", [])
        
        # Validación inicial: verificamos que el último mensaje tenga tool_calls
        last_message = history[-1] if history else None
        if not last_message or not isinstance(last_message, AIMessage):
            logger.warning(f"custom_tool_wrapper - Último mensaje no es AIMessage, saltando ejecución")
            return {"history": history}
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning(f"custom_tool_wrapper - Último mensaje no tiene tool_calls, saltando ejecución")
            return {"history": history}
        
        logger.info(f"custom_tool_wrapper - Ejecutando herramientas con historial de {len(history)} mensajes")
        logger.info(f"custom_tool_wrapper - Tool calls detectados: {len(last_message.tool_calls)}")
        
        tool_messages = []
        
        # Ejecutamos cada herramienta individualmente para mejor control de errores
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get('name', '')
            tool_call_id = tool_call.get('id', '')
            tool_args = tool_call.get('args', {})
            
            logger.info(f"custom_tool_wrapper - Procesando tool_call: name={tool_name}, id={tool_call_id}")
            logger.info(f"custom_tool_wrapper - Args recibidos: {tool_args}")
            
            try:
                # Validación de argumentos
                if not tool_name:
                    raise ValueError("Tool name está vacío")
                
                if not tool_call_id:
                    raise ValueError("Tool call ID está vacío")
                
                # Buscamos la herramienta correspondiente
                tool = next((t for t in tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(f"Herramienta '{tool_name}' no encontrada. Herramientas disponibles: {[t.name for t in tools]}")
                
                # Validamos y normalizamos los argumentos
                # La herramienta search_technical_docs espera un parámetro 'query' de tipo str
                if tool_name == "search_technical_docs":
                    if isinstance(tool_args, dict):
                        query = tool_args.get('query', '')
                        if not query or not isinstance(query, str):
                            # Intentamos extraer el query de diferentes formas
                            query = str(tool_args.get('query', tool_args.get('text', tool_args.get('input', ''))))
                            if not query:
                                raise ValueError(f"Argumento 'query' no encontrado en tool_args: {tool_args}")
                        tool_args = {"query": query}
                    elif isinstance(tool_args, str):
                        tool_args = {"query": tool_args}
                    else:
                        raise ValueError(f"Argumentos inválidos para {tool_name}: {tool_args}")
                
                logger.info(f"custom_tool_wrapper - Ejecutando herramienta '{tool_name}' con args normalizados: {tool_args}")
                
                # Ejecutamos la herramienta con timeout implícito
                try:
                    if hasattr(tool, 'ainvoke'):
                        result = await tool.ainvoke(tool_args)
                    elif hasattr(tool, 'invoke'):
                        result = tool.invoke(tool_args)
                    else:
                        raise ValueError(f"Herramienta '{tool_name}' no tiene métodos ainvoke o invoke")
                    
                    logger.info(f"custom_tool_wrapper - Herramienta '{tool_name}' ejecutada exitosamente")
                    
                except Exception as exec_error:
                    # NO re-lanzamos el error, en su lugar lo manejamos creando un ToolMessage de error
                    logger.error(f"custom_tool_wrapper - Error durante ejecución de '{tool_name}': {str(exec_error)}", exc_info=True)
                    # Creamos un ToolMessage de error en lugar de re-lanzar
                    error_content = json.dumps({
                        "error": "Error ejecutando herramienta",
                        "message": str(exec_error)[:500],
                        "tool_name": tool_name,
                        "error_type": type(exec_error).__name__,
                        "content": f"No se pudo ejecutar la herramienta '{tool_name}'. Por favor, intenta reformular tu pregunta.",
                        "value": []
                    }, ensure_ascii=False)
                    
                    error_msg = ToolMessage(
                        content=error_content,
                        tool_call_id=tool_call_id if tool_call_id else f"error_{len(tool_messages)}"
                    )
                    tool_messages.append(error_msg)
                    logger.warning(f"custom_tool_wrapper - ToolMessage de error creado para '{tool_name}' (en lugar de re-lanzar)")
                    continue  # Continuamos con el siguiente tool_call
                
                # Serializamos el resultado
                if isinstance(result, dict):
                    # Si ya es un dict, lo serializamos a JSON
                    try:
                        content = json.dumps(result, ensure_ascii=False, default=str)
                    except (TypeError, ValueError) as json_error:
                        logger.warning(f"custom_tool_wrapper - Error serializando dict, usando str: {json_error}")
                        content = json.dumps({"content": str(result), "value": []}, ensure_ascii=False)
                elif isinstance(result, str):
                    # Si es string, lo envolvemos en un dict para mantener consistencia
                    content = json.dumps({"content": result, "value": []}, ensure_ascii=False)
                else:
                    # Para otros tipos, convertimos a string y envolvemos
                    content = json.dumps({"content": str(result), "value": []}, ensure_ascii=False)
                
                # Creamos el ToolMessage
                tool_msg = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id
                )
                tool_messages.append(tool_msg)
                logger.info(f"custom_tool_wrapper - ToolMessage creado para '{tool_name}' (content_length={len(content)})")
                
            except Exception as tool_error:
                # SIEMPRE creamos un ToolMessage de error para evitar que el LLM falle
                logger.error(f"custom_tool_wrapper - Error procesando tool_call '{tool_name}': {str(tool_error)}", exc_info=True)
                
                error_content = json.dumps({
                    "error": "Error ejecutando herramienta",
                    "message": str(tool_error)[:500],  # Limitamos el mensaje
                    "tool_name": tool_name,
                    "error_type": type(tool_error).__name__,
                    "content": f"No se pudo ejecutar la herramienta '{tool_name}'. Por favor, intenta reformular tu pregunta.",
                    "value": []
                }, ensure_ascii=False)
                
                error_msg = ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call_id if tool_call_id else f"error_{len(tool_messages)}"
                )
                tool_messages.append(error_msg)
                logger.warning(f"custom_tool_wrapper - ToolMessage de error creado para '{tool_name}'")
        
        # Validación final: garantizamos que TODOS los tool_calls tengan su ToolMessage
        tool_call_ids = {tc.get('id') for tc in last_message.tool_calls if tc.get('id')}
        tool_message_ids = {
            msg.tool_call_id 
            for msg in tool_messages 
            if isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id') and msg.tool_call_id
        }
        
        logger.info(f"custom_tool_wrapper - Tool call IDs esperados: {tool_call_ids}")
        logger.info(f"custom_tool_wrapper - Tool message IDs recibidos: {tool_message_ids}")
        
        missing_ids = tool_call_ids - tool_message_ids
        if missing_ids:
            logger.error(f"custom_tool_wrapper - CRÍTICO: Tool calls sin respuesta: {missing_ids}")
            # Agregamos ToolMessages de error para los faltantes
            for tool_call in last_message.tool_calls:
                if tool_call.get('id') in missing_ids:
                    error_msg = ToolMessage(
                        content=json.dumps({
                            "error": "Error ejecutando herramienta",
                            "message": "No se pudo procesar esta herramienta",
                            "content": "No se pudo ejecutar la herramienta. Por favor, intenta reformular tu pregunta.",
                            "value": []
                        }, ensure_ascii=False),
                        tool_call_id=tool_call.get('id', f"missing_{len(tool_messages)}")
                    )
                    tool_messages.append(error_msg)
            logger.warning(f"custom_tool_wrapper - Agregados {len(missing_ids)} ToolMessages de error para tool_calls faltantes")
        
        logger.info(f"custom_tool_wrapper - Retornando {len(tool_messages)} ToolMessages (esperados {len(last_message.tool_calls)})")
        return {"history": history + tool_messages}
    
    except Exception as wrapper_error:
        # CATCH-ALL: Si algo falla en el wrapper mismo, creamos ToolMessages de error
        logger.critical(f"custom_tool_wrapper - ERROR CRÍTICO en el wrapper: {str(wrapper_error)}", exc_info=True)
        
        history = state.get("history", [])
        last_message = history[-1] if history else None
        
        # Si hay un último mensaje con tool_calls, creamos ToolMessages de error para todos
        if last_message and isinstance(last_message, AIMessage):
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                error_messages = []
                for tool_call in last_message.tool_calls:
                    tool_call_id = tool_call.get('id', f"error_{len(error_messages)}")
                    error_msg = ToolMessage(
                        content=json.dumps({
                            "error": "Error crítico en el sistema",
                            "message": "No se pudo procesar las herramientas correctamente",
                            "content": "Hubo un error al procesar tu solicitud. Por favor, intenta nuevamente.",
                            "value": []
                        }, ensure_ascii=False),
                        tool_call_id=tool_call_id
                    )
                    error_messages.append(error_msg)
                
                logger.warning(f"custom_tool_wrapper - Retornando {len(error_messages)} ToolMessages de error por fallo crítico")
                return {"history": history + error_messages}
        
        # Si no hay tool_calls, devolvemos el historial sin cambios
        logger.warning(f"custom_tool_wrapper - No hay tool_calls, devolviendo historial sin cambios")
        return {"history": history}


async def finalize_node(state: GraphState) -> dict:
    """
    Nodo de finalización que extrae la respuesta final y las fuentes.
    
    ESTRATEGIA ROBUSTA:
    - Busca el último AIMessage sin tool_calls (la respuesta final)
    - Extrae fuentes solo del último turno de conversación (desde el último HumanMessage)
    - Valida que la respuesta sea única y no esté vacía
    """
    history = state.get("history", [])
    current_input = state.get("input", "")
    
    # Buscamos el último AIMessage sin tool_calls (la respuesta final)
    # Nota: tool_calls puede ser None, [] (lista vacía), o una lista con elementos
    last_ai_msg = next(
        (m for m in reversed(history) if isinstance(m, AIMessage) and (not m.tool_calls or len(m.tool_calls) == 0)),
        None,
    )
    
    response_text = last_ai_msg.content if last_ai_msg else ""
    
    # Validación: si no hay respuesta, generamos un mensaje por defecto
    if not response_text or not response_text.strip():
        response_text = "No pude generar una respuesta para tu pregunta. Por favor, intenta reformularla."
        logger.warning(f"finalize_node - No se encontró respuesta, usando mensaje por defecto")
    
    logger.info(f"finalize_node - Respuesta extraída ({len(response_text)} chars)")
    logger.info(f"finalize_node - Respuesta: '{response_text[:100]}...'")
    
    # Extraemos fuentes solo del último turno de conversación
    # Buscamos el último HumanMessage para saber dónde empezó el turno actual
    current_response_sources = []
    seen_keys = set()
    found_last_human = False
    
    # Recorremos el historial desde el final hacia el principio
    for msg in reversed(history):
        # Si encontramos un ToolMessage, extraemos sus fuentes
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

                    # VALIDACIÓN: ¿El nombre del archivo está en el texto de la respuesta?
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
            except Exception as e:
                logger.debug(f"finalize_node - Error extrayendo fuente: {e}")
                continue

        # Al llegar al mensaje del humano, dejamos de buscar herramientas de turnos anteriores
        if isinstance(msg, HumanMessage):
            found_last_human = True
            # Solo procesamos fuentes del último turno (desde este HumanMessage hacia adelante)
            break
    
    logger.info(f"finalize_node - Fuentes encontradas: {len(current_response_sources)}")
    
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
    # Verificamos si el último mensaje tiene tool_calls válidos (no None y no lista vacía)
    if (hasattr(last_message, "tool_calls") and 
        last_message.tool_calls and 
        len(last_message.tool_calls) > 0):
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
