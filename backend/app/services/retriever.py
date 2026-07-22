from __future__ import annotations

from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_filter_builder import QdrantFilterBuilder
from app.services.qdrant_service import QdrantService


class Retriever:
    """Retrieve chunks for a structured SearchRequest.

    Combines vector similarity over the semantic query with structured payload
    filtering. The retriever is intentionally unaware of how the request was
    produced (rule-based or LLM analyzer) and delegates filter translation to a
    QdrantFilterBuilder, so it stays independent of both query parsing and the
    payload layout.
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
