from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult


class Reranker(ABC):
    """Reorder fused results, optionally using the originating request.

    This is the seam for cross-encoder or LLM-based rerankers. Implementations
    receive the request (for access to the query/context) and the fused results,
    and return a reordered list.
    """

    @abstractmethod
    def rerank(
        self, request: SearchRequest, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Return the results reordered by relevance."""
        raise NotImplementedError


class NoOpReranker(Reranker):
    """Reranker that returns results unchanged."""

    def rerank(
        self, request: SearchRequest, results: list[SearchResult]
    ) -> list[SearchResult]:
        return results
