from abc import ABC, abstractmethod
from typing import Any, Dict


class SearchPort(ABC):
    @abstractmethod
    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError
