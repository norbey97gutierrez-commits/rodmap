import logging
from typing import Any, Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient

# Modelos para ESTRUCTURA del índice
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
)

# Modelos para OPERACIONES de búsqueda
from azure.search.documents.models import (
    VectorizedQuery,
)
from langchain_openai import AzureOpenAIEmbeddings

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

    # MÉTODO PARA BÚSQUEDA (RAG)
    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        """Realiza búsqueda híbrida (Vectorial + Texto) recuperando metadatos de página."""

        # Inicialización local para evitar importaciones circulares
        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=settings.AZURE_OPENAI_API_KEY,
        )

        try:
            # Generamos el vector de la pregunta del usuario
            query_vector = await embeddings_model.aembed_query(query)

            async with SearchClient(
                self.endpoint, self.index_name, self.credential
            ) as client:
                # Definimos la consulta vectorial
                vector_query = VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=5, fields="content_vector"
                )

                # Ejecutamos búsqueda Híbrida (Texto + Vectores)
                # IMPORTANTE: Incluimos "page_number" en el select
                results = await client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=5,
                    select=["title", "content", "source", "page_number"],
                )

                context_blocks = []
                sources = []
                async for result in results:
                    # Extraemos el valor real del campo o marcamos como N/A si no existe
                    page = result.get("page_number")
                    page_label = str(page) if page is not None else "N/A"

                    # Construimos un bloque de contexto enriquecido para el LLM
                    context_blocks.append(
                        f"FUENTE: {result['title']}\n"
                        f"METADATOS: Archivo {result['source']}, Página {page_label}\n"
                        f"CONTENIDO: {result['content']}"
                    )

                    # Guardamos la fuente formateada para las etiquetas del frontend
                    if result.get("source"):
                        sources.append(f"{result['source']} (Pág. {page_label})")

                return {
                    "content": "\n\n---\n\n".join(context_blocks),
                    "sources": list(dict.fromkeys(sources)),  # Eliminamos duplicados
                }
        except Exception as e:
            logger.error(f"Error en búsqueda técnica: {e}")
            return {"content": "", "sources": []}

    # MÉTODO PARA INGESTA (CREAMOS EL ÍNDICE)
    async def create_or_update_index(self, index_name: str, vector_dimensions: int):
        """Define la estructura del índice incluyendo campos de metadatos y vectores."""
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
                # Campo crucial para la trazabilidad documental
                SimpleField(name="page_number", type=SearchFieldDataType.Int32),
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

            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
            )
            await self.index_client.create_or_update_index(index)
            logger.info(
                f"Índice '{index_name}' creado con éxito incluyendo campo 'page_number'."
            )
        except Exception as e:
            logger.error(f"Error creando índice: {e}")
            raise e

    # MÉTODO PARA SUBIR VECTORES
    async def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]):
        """Sube los documentos procesados al índice de búsqueda."""
        async with SearchClient(self.endpoint, index_name, self.credential) as client:
            results = await client.upload_documents(documents=vectors)
            return {
                "total_success": sum(1 for r in results if r.succeeded),
                "total_failed": sum(1 for r in results if not r.succeeded),
            }
