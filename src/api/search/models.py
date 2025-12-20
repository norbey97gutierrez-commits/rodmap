from typing import List, Optional

from pydantic import BaseModel


class DocumentModel(BaseModel):
    id: str
    title: str
    content: str
    category: str
    # Este campo almacenará la lista de números (el vector) que genera el LLM
    content_vector: Optional[List[float]] = None
    source: str  # Fundamental para el citado de fuentes
