from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.models.filter import Filter
from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.processors.query_analyzer import QueryAnalyzer
from app.search.fusion import IdentityFusionStrategy, ResultFusionStrategy
from app.search.reranker import NoOpReranker, Reranker
from app.search.strategy import SearchStrategy


@dataclass(frozen=True)
class SearchOutcome:
    """Retrieval results plus any filters fail-open had to drop to find them.

    ``relaxed_filters`` is empty on a normal search; it lists the metadata
    filters removed by the fail-open retry so callers can warn generation that
    the returned sources may not satisfy the requested constraint (e.g. a date).
    """

    results: list[SearchResult]
    relaxed_filters: tuple[Filter, ...] = ()


class SearchEngine:
    """Single public entry point for retrieval, orchestrating a search pipeline.

    Pipeline: query analysis -> strategies (1..N) -> fusion -> reranking.

    The engine owns only orchestration. Query understanding is delegated to an
    injected ``QueryAnalyzer`` (run exactly once to produce a shared
    ``SearchRequest``), backend-specific query/filter translation lives inside
    each ``SearchStrategy``, and merging/ordering are delegated to the injected
    fusion strategy and reranker. The engine has no knowledge of Qdrant, BM25,
    or any storage-specific query language, so new backends are added by
    implementing a ``SearchStrategy`` and registering it here.
    """

    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        strategies: Sequence[SearchStrategy],
        fusion_strategy: ResultFusionStrategy | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("SearchEngine requires at least one strategy")
        self.query_analyzer = query_analyzer
        self.strategies = list(strategies)
        self.fusion_strategy = fusion_strategy or IdentityFusionStrategy()
        self.reranker = reranker or NoOpReranker()

    def search(self, query: str) -> list[SearchResult]:
        """Return ranked results for ``query`` (fail-open, filters dropped silently)."""
        return self.retrieve(query).results

    def retrieve(self, query: str) -> SearchOutcome:
        # Query understanding happens exactly once; every strategy consumes the
        # same backend-agnostic SearchRequest.
        request = self.query_analyzer.analyze(query)
        results = self._retrieve(request)
        # Fail open: a metadata filter that matches nothing (e.g. a mis-resolved
        # date or wrong axis) would otherwise starve generation of context. Retry
        # once without filters so retrieval degrades to semantic search rather
        # than returning an empty set — and report which filters were dropped so
        # generation can flag that the sources may not satisfy them.
        if not results and request.filters:
            results = self._retrieve(replace(request, filters=[]))
            return SearchOutcome(results, tuple(request.filters))
        return SearchOutcome(results)

    def _retrieve(self, request: SearchRequest) -> list[SearchResult]:
        per_strategy_results = [
            strategy.search(request) for strategy in self.strategies
        ]
        fused = self.fusion_strategy.fuse(per_strategy_results)
        return self.reranker.rerank(request, fused)
