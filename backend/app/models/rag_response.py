from dataclasses import dataclass

from app.models.search_result import SearchResult


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[SearchResult]