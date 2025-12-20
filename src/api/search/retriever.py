from langchain_community.vectorstores.azure_ai_search import AzureAISearch
from langchain_openai import AzureOpenAIEmbeddings

from src.api.core.config import settings
from src.api.llm.client import llm

# 1. El "Traductor": Convierte texto en listas de números (Vectores)
# Usamos el modelo que definas en tu Azure (ej: text-embedding-3-small)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",  # Reemplaza con tu nombre de despliegue
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)

# 2. El "Almacén": Conexión vectorial con Azure AI Search
# Este objeto sabe cómo buscar vectores en tu índice
vector_store = AzureAISearch(
    azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
    azure_search_key=settings.AZURE_SEARCH_API_KEY,
    index_name=settings.AZURE_SEARCH_INDEX_NAME,
    embedding_function=embeddings.embed_query,
)


async def rag_query(query: str):
    # 3. Recuperación (Retrieve):
    # Busca los 3 fragmentos (k=3) más similares vectorialmente a la pregunta
    docs = vector_store.similarity_search(query, k=3)

    # 4. Construcción del contexto con "Citas de Fuentes"
    # Incluimos el contenido y el origen para que el LLM pueda citar
    context_list = []
    for i, d in enumerate(docs):
        source_info = f"[Fuente {i + 1}: {d.metadata.get('source', 'Desconocido')}]"
        context_list.append(f"{source_info}\n{d.page_content}")

    context_text = "\n\n---\n\n".join(context_list)

    # 5. Prompt con Grounding y Citación
    prompt = f"""
Eres un asistente técnico experto. Tu tarea es responder la pregunta del usuario basándote UNICAMENTE en el contexto proporcionado.

REGLAS CRÍTICAS:
1. Si la respuesta no está en el contexto, di: "Lo siento, no tengo información suficiente en mis manuales técnicos."
2. Al final de tu respuesta, indica qué fuentes utilizaste (ej: "Fuente 1, Fuente 2").
3. Mantén un tono profesional.

CONTEXTO:
{context_text}

PREGUNTA DEL USUARIO:
{query}
"""

    # 6. Generación (Generate)
    response = await llm.ainvoke(prompt)

    # Devolvemos tanto la respuesta como el contexto (para el State del grafo)
    return {"response": response.content, "context": context_list}
