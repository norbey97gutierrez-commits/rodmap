import logging
import os
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
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

# NUEVA IMPORTACIÓN: Decorador para convertir funciones en herramientas
from langchain.tools import tool

from src.api.core.config import settings

logger = logging.getLogger(__name__)


class AzureAISearchService:
    def __init__(
        self, endpoint: str | None = None, api_key: str | None = None, **kwargs
    ):
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

    # --- MÉTODO PARA EL AGENTE ---
    @tool
    async def search_technical_docs(self, query: str) -> str:
        """
        Busca información técnica sobre Azure (SQL, App Services, Redes, etc.)
        en la base de conocimientos. Úsala cuando el usuario haga preguntas técnicas.
        """
        # Necesitamos generar el embedding para la búsqueda vectorial interna
        # Esto lo haremos dentro de la herramienta para que el Agente solo envíe texto
        from src.api.graph import (
            embeddings_model,  # Importación local para evitar circulares
        )

        query_vector = await embeddings_model.aembed_query(query)

        docs = await self.search(
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            query=query,
            query_vector=query_vector,
            top_k=3,
        )

        if not docs:
            return "No se encontró información relevante en los documentos técnicos."

        return "\n\n".join(
            [f"Doc: {d['title']}\nContenido: {d['content']}" for d in docs]
        )

    # --- MÉTODOS EXISTENTES ---
    async def search(
        self, index_name: str, query: str, query_vector: List[float], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Realiza una búsqueda híbrida (texto + vectores)."""
        async with SearchClient(
            endpoint=self.endpoint, index_name=index_name, credential=self.credential
        ) as search_client:
            vector_query = VectorizedQuery(
                vector=query_vector, k_nearest_neighbors=top_k, fields="content_vector"
            )

            results = await search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["title", "content"],
                top=top_k,
            )

            found_docs = []
            async for result in results:
                found_docs.append(
                    {
                        "title": result.get("title"),
                        "content": result.get("content"),
                        "score": result.get("@search.score"),
                    }
                )
            return found_docs

    # (Mantén aquí tus métodos create_or_update_index y upsert_vectors igual que antes)

    async def create_or_update_index(
        self, index_name: str, vector_dimensions: int = 3072, **kwargs: Any
    ) -> bool:
        try:
            # Definición de campos usando las clases especializadas para evitar errores de atributos
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
                    searchable=True,  # Requerido para vectores
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

            semantic_config = SemanticConfiguration(
                name="default-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="title")],
                ),
            )

            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=SemanticSearch(configurations=[semantic_config]),
            )

            self.index_client.create_or_update_index(index)
            print(f"Éxito: Índice '{index_name}' creado o actualizado.")
            return True
        except Exception as e:
            print(f"Error al crear el índice: {e}")
            return False

    async def upsert_vectors(
        self, index_name: str, vectors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not vectors:
            return {"total_success": 0, "total_failed": 0}

        async with AsyncSearchClient(
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
                print(f"Error en upsert: {e}")
                return {"total_success": 0, "total_failed": len(vectors)}
