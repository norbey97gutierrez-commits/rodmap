"""
RETRIEVER RAG - MOTOR DE BÚSQUEDA Y GENERACIÓN
Este componente une las piezas: busca en la base de datos (Retrieve),
prepara el contexto (Augment) y genera la respuesta final (Generate).
"""

from langchain_community.vectorstores.azure_ai_search import AzureAISearch
from langchain_openai import AzureOpenAIEmbeddings

from src.api.core.config import settings
from src.api.llm.client import llm

# TRADUCTOR DE IDIOMA (MODELO DE EMBEDDINGS)
# Convertimos las palabras del usuario en "coordenadas matemáticas" (vectores).
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)

# ALMACÉN DE CONOCIMIENTO (VECTOR STORE)
# Establecemos la conexión directa con el índice de Azure AI Search.
vector_store = AzureAISearch(
    azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
    azure_search_key=settings.AZURE_SEARCH_API_KEY,
    index_name=settings.AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)


# FUNCIÓN MAESTRA: rag_query
async def rag_query(query: str) -> dict:
    """
    Coordina el flujo completo: busca información, crea el mensaje y obtiene la respuesta.
    """

    # BÚSQUEDA (RETRIEVE)
    # Preguntamos a la base de datos y traemos los 3 párrafos más parecidos a la pregunta.
    docs = vector_store.similarity_search(query, k=3)

    # ARMAMOS EL CONTEXTO
    # Tomamos esos 3 párrafos y les ponemos una "etiqueta" con el nombre del archivo de origen.
    context_list = []
    for i, doc in enumerate(docs):
        source_name = doc.metadata.get("source", "Documento Interno")
        context_list.append(f"[Fuente {i + 1}: {source_name}]\n{doc.page_content}")

    # Unimos todo el texto recuperado en un solo bloque grande
    context_text = "\n\n---\n\n".join(context_list)

    # INSTRUCCIÓN FINAL (PROMPT GROUNDING)
    # Creamos el prompt para evitar que el LLM alucine.
    prompt = f"""
    Eres un asistente técnico experto. Responde EXCLUSIVAMENTE usando la documentación proporcionada abajo.

    DOCUMENTACIÓN DE APOYO:
    {context_text}

    PREGUNTA DEL USUARIO: 
    {query}

    REGLAS DE ORO:
    1. Si la respuesta no está en la documentación, di que no tienes esa información.
    2. Cita siempre el nombre del archivo fuente (ej. [Fuente 1: Manual.pdf]).
    3. Mantén un tono profesional y directo.
    """

    # GENERACIÓN
    # Le enviamos el mensaje completo al modelo de lenguaje
    response = await llm.ainvoke(prompt)

    # Retornamos un diccionario limpio para que el Frontend lo use fácilmente
    return {
        "response": response.content,  # El texto de la respuesta
        "context": context_list,  # El material que usamos (documents.json)
    }
