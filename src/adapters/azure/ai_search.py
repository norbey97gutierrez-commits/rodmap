import logging
from typing import Any, Dict, List

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
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings

from src.domain.ports.search_port import SearchPort
from src.infrastructure.azure_setup import settings

logger = logging.getLogger(__name__)


class AzureAISearchService(SearchPort):
    def __init__(self):
        self.endpoint = str(settings.AZURE_SEARCH_ENDPOINT)
        self.api_key = settings.AZURE_SEARCH_API_KEY
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )

    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        logger.info(f"search_technical_docs - Iniciando búsqueda para query: '{query[:100]}...'")

        embeddings_model = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=settings.AZURE_OPENAI_API_KEY,
        )

        try:
            if not self.endpoint or not self.index_name or not self.credential:
                raise ValueError(
                    "Configuración de Azure Search incompleta. Verifica las variables de entorno."
                )

            logger.info("search_technical_docs - Generando embeddings...")
            query_vector = await embeddings_model.aembed_query(query)
            logger.info(
                f"search_technical_docs - Embeddings generados, dimensión: {len(query_vector)}"
            )

            async with SearchClient(
                self.endpoint, self.index_name, self.credential
            ) as client:
                vector_query = VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=5, fields="content_vector"
                )

                logger.info("search_technical_docs - Ejecutando búsqueda híbrida...")
                results = await client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=5,
                    select=["title", "content", "source", "page_number"],
                )

                context_blocks = []
                raw_docs = []

                result_count = 0
                async for result in results:
                    result_count += 1
                    page = result.get("page_number")
                    page_label = str(page) if page is not None else "N/A"

                    context_blocks.append(
                        f"FUENTE: {result.get('title', 'Sin título')}\n"
                        f"METADATOS: Archivo {result.get('source')}, Página {page_label}\n"
                        f"CONTENIDO: {result.get('content')}"
                    )

                    raw_docs.append(
                        {
                            "source": result.get("source"),
                            "page_number": page,
                            "title": result.get("title"),
                            "url": result.get("url") or "#",
                        }
                    )

                logger.info(
                    f"search_technical_docs - Búsqueda completada, {result_count} resultados encontrados"
                )

                result_dict = {
                    "content": "\n\n---\n\n".join(context_blocks) if context_blocks else "",
                    "value": raw_docs,
                }

                logger.info(
                    "search_technical_docs - Retornando resultado: "
                    f"content_length={len(result_dict['content'])}, "
                    f"docs_count={len(result_dict['value'])}"
                )
                return result_dict

        except Exception as e:
            logger.error(f"search_technical_docs - Error en búsqueda técnica: {e}", exc_info=True)
            logger.error(f"search_technical_docs - Tipo de error: {type(e).__name__}")
            error_result = {
                "content": f"Error al buscar documentos: {str(e)[:200]}",
                "value": [],
            }
            logger.warning("search_technical_docs - Retornando resultado de error")
            return error_result

    async def create_or_update_index(self, index_name: str, vector_dimensions: int):
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
            logger.info(f"Índice '{index_name}' actualizado.")
        except Exception as e:
            logger.error(f"Error creando índice: {e}")
            raise e

    async def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]):
        async with SearchClient(self.endpoint, index_name, self.credential) as client:
            results = await client.upload_documents(documents=vectors)
            return {
                "total_success": sum(1 for r in results if r.succeeded),
                "total_failed": sum(1 for r in results if not r.succeeded),
            }
