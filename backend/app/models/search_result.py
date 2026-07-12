from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float