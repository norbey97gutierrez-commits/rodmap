import asyncio
import json

from langchain_openai import AzureOpenAIEmbeddings

from src.api.core.config import settings
from src.api.search.service import AzureAISearchService

# 1. Inicializamos el modelo de embeddings
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)


async def index_documents():
    print("--- Iniciando proceso de indexación profesional ---")

    # 2. Instanciamos el servicio profesional
    search_service = AzureAISearchService(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        api_key=settings.AZURE_SEARCH_API_KEY,
        embedding_model=embeddings_model,
    )

    # 3. CREACIÓN AUTOMÁTICA DEL ÍNDICE
    # Esto elimina la necesidad de ir al portal de Azure manualmente
    print(f"Verificando/Creando índice: {settings.AZURE_SEARCH_INDEX_NAME}...")
    await search_service.create_or_update_index(
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        vector_dimensions=3072,  # Dimensión estándar de OpenAI
    )

    # 4. Cargar datos del JSON local
    try:
        with open("data/documents.json", "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        print("Error: No se encontró el archivo data/documents.json")
        return

    print(f"Procesando {len(documents)} documentos...")

    processed_docs = []
    for doc in documents:
        print(f"-> Generando vector para: {doc['title']}")

        # Generamos el vector usando el modelo
        vector = embeddings_model.embed_query(doc["content"])

        # Mapeamos los campos del JSON a lo que espera el servicio
        # El servicio profesional espera: id, title, content, category, source, content_vector
        doc["content_vector"] = vector

        # Agregamos campos extra si el servicio de tu compañero los requiere
        # (Asegúrate de que coincidan con los 'fields' que definimos antes)
        processed_docs.append(doc)

    # 5. SUBIDA PROFESIONAL (UPSERT)
    # El servicio de tu compañero ya maneja reintentos y batches
    print("Subiendo documentos a Azure...")
    stats = await search_service.upsert_vectors(
        index_name=settings.AZURE_SEARCH_INDEX_NAME, vectors=processed_docs
    )

    print("--- Proceso finalizado ---")
    print(f"Exitosos: {stats['total_success']}")
    print(f"Fallidos: {stats['total_failed']}")


if __name__ == "__main__":
    # Como el servicio es asíncrono (async), usamos asyncio.run
    asyncio.run(index_documents())
