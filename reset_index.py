# reset_index.py
import asyncio

from src.api.core.config import settings
from src.api.search.service import AzureAISearchService


async def reset():
    service = AzureAISearchService(
        endpoint=settings.AZURE_SEARCH_ENDPOINT, api_key=settings.AZURE_SEARCH_API_KEY
    )
    # Esto borra el índice corrupto
    await service.delete_index(settings.AZURE_SEARCH_INDEX_NAME)


if __name__ == "__main__":
    asyncio.run(reset())
