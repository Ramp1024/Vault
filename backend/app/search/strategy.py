from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.services.bm25_index import BM25Index, RankBM25Index
from app.services.bm25_query_builder import BM25QueryBuilder
from app.services.embedding_service import EmbeddingService
from app.services.metadata_filter_matcher import MetadataFilterMatcher
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


class BM25SearchStrategy(SearchStrategy):
    """Lexical (BM25) retrieval over the persisted BM25 index.

    Owns everything specific to BM25: translating the request into a term query
    (via ``BM25QueryBuilder``), executing the lexical search against the injected
    ``BM25Index`` abstraction, and applying metadata filters after retrieval
    (BM25 has no native filtering) so behavior matches vector retrieval. All BM25
    knowledge lives here and in its collaborators, keeping ``SearchEngine``
    unaware of BM25.

    The persisted index is loaded lazily on first use so the process does not
    rebuild it on startup; an already-built index can also be injected directly.
    """

    def __init__(
        self,
        index: BM25Index | None = None,
        query_builder: BM25QueryBuilder | None = None,
        filter_matcher: MetadataFilterMatcher | None = None,
    ) -> None:
        self.index = index or RankBM25Index()
        self.query_builder = query_builder or BM25QueryBuilder()
        self.filter_matcher = filter_matcher or MetadataFilterMatcher()

    def search(self, request: SearchRequest) -> list[SearchResult]:
        self._ensure_index_ready()

        query_tokens = self.query_builder.build(request)
        if not query_tokens:
            return []

        # Retrieve the full ranked candidate set so post-retrieval metadata
        # filtering cannot discard results that belong in the final top_k.
        candidates = self.index.search(query_tokens, limit=None)

        if request.filters:
            candidates = [
                result
                for result in candidates
                if self.filter_matcher.matches(result.chunk, request.filters)
            ]

        return candidates[: request.top_k]

    def _ensure_index_ready(self) -> None:
        """Load the persisted index on first use unless one is already in memory."""
        if self.index.size == 0 and self.index.exists():
            self.index.load()
