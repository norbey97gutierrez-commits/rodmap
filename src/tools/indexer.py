"""
INDEXADOR DE DOCUMENTOS PARA AZURE AI SEARCH
Script para crear/actualizar índices y cargar documentos con embeddings vectoriales.
"""

import asyncio
import json
import logging
from pathlib import Path

from langchain_openai import AzureOpenAIEmbeddings

from src.api.core.config import settings
from src.api.search.service import AzureAISearchService

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DEL MODELO DE EMBEDDINGS
# ============================================================================
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
    api_key=settings.AZURE_OPENAI_API_KEY,
    chunk_size=16,
    max_retries=3,
    timeout=30.0,
)


# ============================================================================
# FUNCIÓN PRINCIPAL DE INDEXACIÓN
# ============================================================================
async def index_documents():
    """
    Flujo completo de indexación profesional.
    """
    print("=" * 60)
    print("🚀 INICIANDO PROCESO DE INDEXACIÓN PROFESIONAL")
    print("=" * 60)

    # 1. Inicialización
    search_service = AzureAISearchService()
    data_file = Path("data/documents.json")

    if not data_file.exists():
        print(f"❌ Error: No se encontró el archivo {data_file}")
        return

    # 2. Limpieza y Creación de Índice
    print(f"📋 Preparando índice: {settings.AZURE_SEARCH_INDEX_NAME}")
    try:
        # Borramos para asegurar que el esquema (id, campos semánticos) sea el nuevo
        print("   🗑️ Eliminando índice antiguo para actualizar esquema...")
        try:
            await search_service.index_client.delete_index(
                settings.AZURE_SEARCH_INDEX_NAME
            )
        except Exception:
            pass

        await search_service.create_or_update_index(
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            vector_dimensions=3072,  # text-embedding-3-large
        )
        print("   ✅ Índice recreado exitosamente")
    except Exception as e:
        print(f"   ❌ Error crítico en infraestructura: {e}")
        return

    # 3. Carga de datos
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"📄 Cargados {len(documents)} documentos desde JSON")
    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")
        return

    # 4. Procesamiento y Embeddings
    print("🔧 Generando embeddings y preparando paquetes...")
    processed_docs = []

    for i, doc in enumerate(documents, 1):
        try:
            doc_id = str(doc.get("id", f"doc-{i:03d}"))
            title = doc.get("title", "Sin título")

            # Combinamos título y contenido para un vector más descriptivo
            text_to_embed = f"{title}: {doc.get('content', '')}"

            # Generación asíncrona del vector
            vector = await embeddings_model.aembed_query(text_to_embed)

            processed_docs.append(
                {
                    "id": doc_id,
                    "title": title,
                    "content": doc.get("content", ""),
                    "content_vector": vector,
                    "source": doc.get("source", "manual-ingest"),
                    "category": doc.get("category", "General"),
                }
            )

            if i % 5 == 0:
                print(f"   📊 Progreso: {i}/{len(documents)} procesados")

        except Exception as e:
            print(f"   ⚠️ Error en doc {i}: {e}")

    # 5. Subida a Azure
    print(f"⬆️  Subiendo {len(processed_docs)} vectores a Azure AI Search...")
    try:
        stats = await search_service.upsert_vectors(
            index_name=settings.AZURE_SEARCH_INDEX_NAME, vectors=processed_docs
        )

        print("=" * 60)
        print("📊 RESUMEN DE INDEXACIÓN")
        print("=" * 60)
        print(f"   Éxitos: {stats.get('total_success')}")
        print(f"   Fallos: {stats.get('total_failed')}")
        print("\n✅ Proceso Finalizado")

    except Exception as e:
        print(f"❌ Error en la subida: {e}")

    finally:
        await search_service.index_client.close()


if __name__ == "__main__":
    asyncio.run(index_documents())
