from __future__ import annotations

from collections.abc import Sequence

from app.models.search_result import SearchResult
from app.processors.query_analyzer import QueryAnalyzer
from app.search.fusion import IdentityFusionStrategy, ResultFusionStrategy
from app.search.reranker import NoOpReranker, Reranker
from app.search.strategy import SearchStrategy


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
        # Query understanding happens exactly once; every strategy consumes the
        # same backend-agnostic SearchRequest.
        request = self.query_analyzer.analyze(query)
        per_strategy_results = [
            strategy.search(request) for strategy in self.strategies
        ]
        fused = self.fusion_strategy.fuse(per_strategy_results)
        return self.reranker.rerank(request, fused)
