import asyncio
import json
import logging
import os

from src.api.core.config import settings
from src.api.graph import embeddings_model
from src.api.search.service import AzureAISearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_json_data(file_path: str):
    search_service = AzureAISearchService()

    # 1. Leer el archivo JSON
    logger.info(f"📂 Cargando archivo: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"❌ El archivo {file_path} no existe.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # 2. LIMPIEZA: Eliminar índice antiguo para evitar errores de cambio de esquema
    logger.info(
        f"🗑️ Eliminando índice antiguo '{settings.AZURE_SEARCH_INDEX_NAME}' si existe..."
    )
    try:
        await search_service.index_client.delete_index(settings.AZURE_SEARCH_INDEX_NAME)
        logger.info("✅ Índice antiguo eliminado.")
    except Exception:
        logger.info("ℹ️ El índice no existía o ya estaba limpio. Continuando...")

    # 3. CREACIÓN: Crear el índice con la estructura vectorial y semántica correcta
    logger.info("🛠️ Creando nuevo índice en Azure...")
    await search_service.create_or_update_index(
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        vector_dimensions=3072,  # Ajustado para text-embedding-3-large
    )

    # 4. GENERACIÓN: Crear Embeddings
    logger.info(f"🧠 Generando embeddings para {len(documents)} documentos...")
    processed_docs = []

    for doc in documents:
        text_to_embed = f"{doc['title']}: {doc['content']}"
        embedding = await embeddings_model.aembed_query(text_to_embed)

        processed_docs.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "category": doc["category"],
                "source": doc["source"],
                "content_vector": embedding,
            }
        )

    # 5. CARGA: Subir vectores a Azure AI Search
    logger.info("📤 Subiendo vectores a Azure...")
    result = await search_service.upsert_vectors(
        index_name=settings.AZURE_SEARCH_INDEX_NAME, vectors=processed_docs
    )

    logger.info(
        f"✅ Ingesta completada: {result['total_success']} éxitos, {result['total_failed']} fallos."
    )

    # Cerrar el cliente para evitar errores de "Unclosed client session"
    await search_service.index_client.close()


if __name__ == "__main__":
    # Asegúrate de que la ruta sea correcta según tu estructura
    asyncio.run(ingest_json_data("data/documents.json"))
