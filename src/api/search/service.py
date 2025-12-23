import logging
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
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
from azure.search.documents.models import VectorizedQuery

from src.api.core.config import settings

logger = logging.getLogger(__name__)


class AzureAISearchService:
    def __init__(
        self, endpoint: Optional[str] = None, api_key: Optional[str] = None, **kwargs
    ):
        self.endpoint = endpoint or settings.AZURE_SEARCH_ENDPOINT
        self.api_key = api_key or settings.AZURE_SEARCH_API_KEY
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        """Busca información técnica y devuelve contenido + fuentes."""
        from src.api.graph import embeddings_model

        try:
            # 1. Generar embedding
            query_vector = await embeddings_model.aembed_query(query)

            # 2. Llamada al método interno self.search
            docs = await self.search(
                index_name=settings.AZURE_SEARCH_INDEX_NAME,
                query=query,
                query_vector=query_vector,
                top_k=3,
            )

            if not docs:
                return {"content": "No se encontró información.", "sources": []}

            # 3. Formatear respuesta
            context_parts = []
            sources = []
            for d in docs:
                context_parts.append(
                    f"DOCUMENTO: {d['title']}\nCONTENIDO: {d['content']}"
                )
                if d.get("source"):
                    sources.append(d["source"])

            return {
                "content": "\n\n---\n\n".join(context_parts),
                "sources": list(set(sources)),
            }
        except Exception as e:
            logger.error(f"Error en search_technical_docs: {e}")
            return {"content": f"Error: {str(e)}", "sources": []}

    async def search(
        self, index_name: str, query: str, query_vector: List[float], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Método core de búsqueda (Ahora correctamente indentado dentro de la clase)."""
        async with SearchClient(
            endpoint=self.endpoint, index_name=index_name, credential=self.credential
        ) as search_client:
            vector_query = VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
            )

            results = await search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["title", "content", "source"],
                top=top_k,
            )

            found_docs = []
            async for result in results:
                score = result.get("@search.score")
                found_docs.append(
                    {
                        "title": result.get("title"),
                        "content": result.get("content"),
                        "source": result.get("source"),
                        "score": score,
                    }
                )
                # Log para ver la relevancia en la terminal
                print(f"[DEBUG] Documento: {result.get('title')} | Score: {score}")

            return found_docs

    async def create_or_update_index(
        self, index_name: str, vector_dimensions: int = 3072
    ) -> bool:
        """Define la estructura del índice en Azure."""
        try:
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="title", type=SearchFieldDataType.String, retrievable=True
                ),
                SearchableField(
                    name="content", type=SearchFieldDataType.String, retrievable=True
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimensions,
                    vector_search_profile_name="default-vector-profile",
                ),
                SimpleField(
                    name="category",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="source",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    retrievable=True,
                ),
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
            return True
        except Exception as e:
            logger.error(f"Error creando índice: {e}")
            return False

    async def upsert_vectors(
        self, index_name: str, vectors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sube documentos al índice."""
        async with SearchClient(
            endpoint=self.endpoint, index_name=index_name, credential=self.credential
        ) as search_client:
            try:
                results = await search_client.upload_documents(documents=vectors)
                success_count = sum(1 for r in results if r.succeeded)
                return {
                    "total_success": success_count,
                    "total_failed": len(vectors) - success_count,
                }
            except Exception as e:
                logger.error(f"Error en upsert_vectors: {e}")
                return {"total_success": 0, "total_failed": len(vectors)}
