"""
RETRIEVER RAG PARA AZURE AI SEARCH
Sistema de búsqueda y recuperación de documentos con embeddings vectoriales.
Componente clave del flujo RAG (Retrieval Augmented Generation).
"""

from langchain_community.vectorstores.azure_ai_search import AzureAISearch
from langchain_openai import AzureOpenAIEmbeddings

from src.api.core.config import settings
from src.api.llm.client import llm

# ============================================================================
# 1. MODELO DE EMBEDDINGS (CODIFICACIÓN DE TEXTO A VECTORES)
# ============================================================================
"""
Convierte texto en vectores numéricos para búsqueda semántica.
Modelo: text-embedding-3-large (3072 dimensiones)
"""
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)


# ============================================================================
# 2. VECTOR STORE (ALMACÉN DE VECTORES EN AZURE AI SEARCH)
# ============================================================================
"""
Conexión al índice de Azure AI Search para búsquedas vectoriales.
Index: Configurado previamente con esquema de vectores (3072D).
"""
vector_store = AzureAISearch(
    azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
    azure_search_key=settings.AZURE_SEARCH_API_KEY,
    index_name=settings.AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)


# ============================================================================
# 3. FUNCIÓN PRINCIPAL DE CONSULTA RAG
# ============================================================================
async def rag_query(query: str) -> dict:
    """
    Ejecuta el flujo completo RAG: Retrieve + Augment + Generate.

    Args:
        query: Pregunta del usuario

    Returns:
        dict con respuesta generada y contexto usado
    """
    # ----- FASE 1: RECUPERACIÓN (RETRIEVE) -----
    # Búsqueda por similitud semántica (k=3 resultados más relevantes)
    docs = vector_store.similarity_search(query, k=3)

    # ----- FASE 2: CONSTRUCCIÓN DE CONTEXTO -----
    # Formatea documentos con fuentes para trazabilidad
    context_list = []
    for i, doc in enumerate(docs):
        source = f"[Fuente {i + 1}: {doc.metadata.get('source', 'Desconocido')}]"
        context_list.append(f"{source}\n{doc.page_content}")

    context_text = "\n\n---\n\n".join(context_list)

    # ----- FASE 3: GENERACIÓN CON GROUNDING -----
    # Prompt que fuerza al LLM a usar solo el contexto proporcionado
    prompt = f"""
Basándote EXCLUSIVAMENTE en esta documentación:

{context_text}

Responde esta pregunta: {query}

Reglas:
1. Solo usa información del contexto
2. Cita las fuentes al final
3. Si no hay información, indícalo
"""

    # ----- FASE 4: GENERACIÓN DE RESPUESTA -----
    response = await llm.ainvoke(prompt)

    return {
        "response": response.content,  # Respuesta generada
        "context": context_list,  # Fuentes usadas (para trazabilidad)
    }
