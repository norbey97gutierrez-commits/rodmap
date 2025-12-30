import logging
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

# 1. Modelos para ESTRUCTURA del índice (Creación)
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

# 2. Modelos para OPERACIONES de búsqueda (Ejecución)
from azure.search.documents.models import (
    QueryType,  # <-- QueryType se importa de aquí, no de indexes
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

    # --- MÉTODO PARA BÚSQUEDA (RAG) ---
    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        """Realiza búsqueda híbrida y semántica en el índice."""
        from src.api.graph import embeddings_model

        try:
            query_vector = await embeddings_model.aembed_query(query)
            async with SearchClient(
                self.endpoint, self.index_name, self.credential
            ) as client:
                vector_query = VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=5, fields="content_vector"
                )

                results = await client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    query_type=QueryType.SEMANTIC,
                    semantic_configuration_name="default-semantic-config",
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
            logger.error(f"Error en búsqueda: {e}")
            return {"content": "", "sources": []}

    # --- MÉTODO PARA INGESTA (CREAR ÍNDICE) ---
    async def create_or_update_index(self, index_name: str, vector_dimensions: int):
        """Define la estructura del índice, incluyendo vectores y semántica."""
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

            semantic_search = SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="default-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[SemanticField(field_name="content")],
                            keywords_fields=[SemanticField(field_name="title")],
                        ),
                    )
                ]
            )

            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search,
            )
            await self.index_client.create_or_update_index(index)
            logger.info(f"✅ Índice '{index_name}' creado/actualizado.")
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
