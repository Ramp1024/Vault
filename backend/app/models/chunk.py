from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    content: str
    chunk_index: int
    metadata: dict[str, Any]
