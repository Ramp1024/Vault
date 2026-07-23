from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_filter_builder import QdrantFilterBuilder
from app.services.qdrant_service import QdrantService


class SearchStrategy(ABC):
    """A single retrieval technique that turns a SearchRequest into results.

    Implementations own everything specific to their technique (embedding,
    filter translation, backend calls) and return backend-agnostic
    ``SearchResult`` objects. New techniques (BM25, hybrid, etc.) are added by
    implementing this interface and registering the strategy with the
    ``SearchEngine`` — no existing strategy needs to change.
    """

    @abstractmethod
    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Return results for the given request."""
        raise NotImplementedError


class VectorSearchStrategy(SearchStrategy):
    """Dense vector retrieval over Qdrant.

    Owns embedding generation for the semantic query, translation of the
    request's structured filters into a Qdrant payload filter, and the vector
    search itself. All Qdrant-specific knowledge lives here (and in the injected
    ``QdrantFilterBuilder``), keeping the ``SearchEngine`` storage-agnostic.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
        filter_builder: QdrantFilterBuilder | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService(get_qdrant_client())
        self.filter_builder = filter_builder or QdrantFilterBuilder()

    def search(self, request: SearchRequest) -> list[SearchResult]:
        semantic_query = request.semantic_query.strip()
        query_embedding = (
            self.embedding_service.embed(semantic_query) if semantic_query else []
        )
        query_filter = self.filter_builder.build(request.filters)

        return self.qdrant_service.search(
            query_embedding=query_embedding,
            limit=request.top_k,
            query_filter=query_filter,
        )
