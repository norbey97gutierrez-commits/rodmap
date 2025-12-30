import logging
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

# 1. Modelos para ESTRUCTURA del índice
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    # Eliminamos SemanticSearch y derivados para simplificar
)

# 2. Modelos para OPERACIONES de búsqueda
from azure.search.documents.models import (
    VectorizedQuery,
)

from src.api.core.config import settings

logger = logging.getLogger(__name__)


class AzureAISearchService:
    def __init__(self):
        self.endpoint = str(settings.AZURE_SEARCH_ENDPOINT)
        self.api_key = settings.AZURE_SEARCH_API_KEY
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

    # --- MÉTODO PARA BÚSQUEDA (RAG) ACTUALIZADO ---
    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        """Realiza búsqueda híbrida (Vectorial + Texto) sin dependencia semántica."""
        from src.api.graph import embeddings_model

        try:
            # Generamos el vector de la consulta del usuario
            query_vector = await embeddings_model.aembed_query(query)

            async with SearchClient(
                self.endpoint, self.index_name, self.credential
            ) as client:
                # Definimos la consulta vectorial
                vector_query = VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=5, fields="content_vector"
                )

                # Ejecutamos búsqueda Híbrida (Texto + Vectores)
                # Eliminamos query_type=SEMANTIC y semantic_configuration_name
                results = await client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=5,
                    select=["title", "content", "source"],
                )

                context_blocks = []
                sources = []
                async for result in results:
                    context_blocks.append(
                        f"FUENTE: {result['title']}\nCONTENIDO: {result['content']}"
                    )
                    if result.get("source"):
                        sources.append(result["source"])

                return {
                    "content": "\n\n---\n\n".join(context_blocks),
                    "sources": list(dict.fromkeys(sources)),
                }
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            return {"content": "", "sources": []}

    # --- MÉTODO PARA INGESTA (CREAR ÍNDICE) ---
    async def create_or_update_index(self, index_name: str, vector_dimensions: int):
        """Define la estructura del índice enfocada en vectores."""
        try:
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=vector_dimensions,
                    vector_search_profile_name="default-vector-profile",
                ),
                SimpleField(name="category", type=SearchFieldDataType.String),
                SimpleField(name="source", type=SearchFieldDataType.String),
            ]

            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw", parameters=HnswParameters(metric="cosine")
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw",
                    )
                ],
            )

            # Creamos el índice sin la configuración semántica para evitar errores de compatibilidad
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
            )
            await self.index_client.create_or_update_index(index)
            logger.info(
                f"✅ Índice '{index_name}' creado/actualizado para búsqueda vectorial."
            )
        except Exception as e:
            logger.error(f"Error creando índice: {e}")
            raise e

    # --- MÉTODO PARA SUBIR VECTORES ---
    async def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]):
        """Sube los documentos procesados con sus embeddings al índice."""
        async with SearchClient(self.endpoint, index_name, self.credential) as client:
            results = await client.upload_documents(documents=vectors)
            return {
                "total_success": sum(1 for r in results if r.succeeded),
                "total_failed": sum(1 for r in results if not r.succeeded),
            }
