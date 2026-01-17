import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.domain.ports.search_port import SearchPort

logger = logging.getLogger(__name__)


class LocalJsonSearchService(SearchPort):
    def __init__(self, data_path: str = "data/documents.json"):
        self.data_path = Path(data_path)

    async def search_technical_docs(self, query: str) -> Dict[str, Any]:
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"No existe {self.data_path}")

            docs = json.loads(self.data_path.read_text(encoding="utf-8"))
            terms = [t.lower() for t in query.split() if t.strip()]
            matches: List[Dict[str, Any]] = []

            for doc in docs:
                haystack = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
                if any(term in haystack for term in terms):
                    matches.append(doc)

            context_blocks = []
            for doc in matches[:5]:
                page = doc.get("page_number")
                page_label = str(page) if page is not None else "N/A"
                context_blocks.append(
                    "FUENTE: {title}\n"
                    "METADATOS: Archivo {source}, Página {page}\n"
                    "CONTENIDO: {content}".format(
                        title=doc.get("title", "Sin título"),
                        source=doc.get("source"),
                        page=page_label,
                        content=doc.get("content"),
                    )
                )

            return {
                "content": "\n\n---\n\n".join(context_blocks) if context_blocks else "",
                "value": [
                    {
                        "source": doc.get("source"),
                        "page_number": doc.get("page_number"),
                        "title": doc.get("title"),
                        "url": doc.get("url") or "#",
                    }
                    for doc in matches[:5]
                ],
            }

        except Exception as e:
            logger.error(f"LocalJsonSearchService error: {e}", exc_info=True)
            return {"content": f"Error local search: {str(e)[:200]}", "value": []}
