from __future__ import annotations

from enum import Enum

from app.processors.query_analyzer import QueryAnalyzer
from app.search.engine import SearchEngine
from app.search.fusion import IdentityFusionStrategy, ReciprocalRankFusion
from app.search.reranker import Reranker
from app.search.strategy import (
    BM25SearchStrategy,
    SearchStrategy,
    VectorSearchStrategy,
)

DEFAULT_RRF_K = 60


class RetrievalMode(str, Enum):
    """Selectable retrieval configurations for the search pipeline."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


def build_search_engine(
    mode: RetrievalMode,
    query_analyzer: QueryAnalyzer,
    *,
    vector_strategy: VectorSearchStrategy | None = None,
    bm25_strategy: BM25SearchStrategy | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    reranker: Reranker | None = None,
) -> SearchEngine:
    """Assemble a ``SearchEngine`` for the requested retrieval mode.

    This is the single place that maps a mode onto a concrete set of strategies
    and a fusion strategy, so callers select Vector / BM25 / Hybrid purely
    through configuration — the ``SearchEngine`` API is untouched. Strategy
    instances may be injected to share expensive collaborators (embedding
    service, BM25 index) across engines.

    Single-strategy modes use identity fusion (native ranking preserved); hybrid
    combines both strategies with Reciprocal Rank Fusion. Adding a future
    strategy to hybrid requires only extending the strategy list here — the
    fusion algorithm stays the same.

    An optional ``reranker`` (e.g. a cross-encoder) is attached as the final
    pipeline stage. Strategies and fusion remain unaware of it: the reranker only
    reorders whatever fused results it receives.
    """
    if mode == RetrievalMode.VECTOR:
        strategies: list[SearchStrategy] = [vector_strategy or VectorSearchStrategy()]
        fusion = IdentityFusionStrategy()
    elif mode == RetrievalMode.BM25:
        strategies = [bm25_strategy or BM25SearchStrategy()]
        fusion = IdentityFusionStrategy()
    elif mode == RetrievalMode.HYBRID:
        strategies = [
            vector_strategy or VectorSearchStrategy(),
            bm25_strategy or BM25SearchStrategy(),
        ]
        fusion = ReciprocalRankFusion(k=rrf_k)
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    return SearchEngine(
        query_analyzer=query_analyzer,
        strategies=strategies,
        fusion_strategy=fusion,
        reranker=reranker,
    )
