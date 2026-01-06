"""
INDEXADOR DE DOCUMENTOS PARA AZURE AI SEARCH
Script actualizado para incluir metadatos de página y limpieza de esquema.
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

# CONFIGURACIÓN DEL MODELO DE EMBEDDINGS
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
    api_key=settings.AZURE_OPENAI_API_KEY,
    chunk_size=16,
    max_retries=3,
    timeout=30.0,
)


async def index_documents():
    """
    Flujo de indexación: Limpia el índice, genera vectores y sube metadatos (incluyendo páginas).
    """
    print("=" * 60)
    print("INICIANDO PROCESO DE INDEXACIÓN PROFESIONAL (V2 - CON PÁGINAS)")
    print("=" * 60)

    search_service = AzureAISearchService()
    # Asegúrate de que la ruta sea correcta desde la raíz del proyecto
    data_file = Path("data/documents.json")

    if not data_file.exists():
        print(f"❌ Error: No se encontró el archivo {data_file}")
        return

    # 1. LIMPIEZA Y RECREACIÓN DEL ÍNDICE
    print(f"Preparando índice: {settings.AZURE_SEARCH_INDEX_NAME}")
    try:
        print("Eliminando índice antiguo para actualizar el esquema (page_number)...")
        try:
            await search_service.index_client.delete_index(
                settings.AZURE_SEARCH_INDEX_NAME
            )
        except Exception:
            print("No existía un índice previo, creando uno nuevo.")

        # Recreamos el índice con las dimensiones del modelo Large (3072)
        await search_service.create_or_update_index(
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            vector_dimensions=3072,
        )
        print("✅ Índice recreado exitosamente con nuevo esquema.")
    except Exception as e:
        print(f"❌ Error crítico en infraestructura: {e}")
        return

    # 2. CARGA DE DATOS LOCALES
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"📂 Cargados {len(documents)} documentos desde JSON")
    except Exception as e:
        print(f"❌ Error leyendo JSON: {e}")
        return

    # 3. PROCESAMIENTO Y GENERACIÓN DE EMBEDDINGS
    print("🧠 Generando embeddings y preparando documentos...")
    processed_docs = []

    for i, doc in enumerate(documents, 1):
        try:
            doc_id = str(doc.get("id", f"doc-{i:03d}"))
            title = doc.get("title", "Sin título")

            # Combinamos título y contenido para mejorar la calidad del vector
            text_to_embed = f"{title}: {doc.get('content', '')}"
            vector = await embeddings_model.aembed_query(text_to_embed)

            # --- MAPEO DE CAMPOS HACIA AZURE ---
            processed_docs.append(
                {
                    "id": doc_id,
                    "title": title,
                    "content": doc.get("content", ""),
                    "content_vector": vector,
                    "source": doc.get("source", "manual-ingest"),
                    "category": doc.get("category", "General"),
                    # NUEVO: Captura el número de página, por defecto 0 si no existe
                    "page_number": doc.get("page_number", 0),
                }
            )

            if i % 2 == 0:
                print(f"  > Progreso: {i}/{len(documents)} procesados")

        except Exception as e:
            print(f"⚠️ Error en doc {i}: {e}")

    # 4. SUBIDA A AZURE AI SEARCH
    print(f"🚀 Subiendo {len(processed_docs)} vectores a Azure...")
    try:
        stats = await search_service.upsert_vectors(
            index_name=settings.AZURE_SEARCH_INDEX_NAME, vectors=processed_docs
        )

        print("=" * 60)
        print("RESUMEN DE INDEXACIÓN")
        print("=" * 60)
        print(f"   Éxitos: {stats.get('total_success')}")
        print(f"   Fallos: {stats.get('total_failed')}")
        print("\n✅ Proceso Finalizado Correctamente")

    except Exception as e:
        print(f"❌ Error en la subida: {e}")

    finally:
        await search_service.index_client.close()


if __name__ == "__main__":
    asyncio.run(index_documents())
