from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.search_result import SearchResult


class ResultFusionStrategy(ABC):
    """Combine result lists produced by multiple search strategies into one list.

    Fusion is where techniques like reciprocal-rank fusion or weighted merging
    will live. It receives one result list per strategy (in strategy order) and
    returns a single ordered list.
    """

    @abstractmethod
    def fuse(self, results: list[list[SearchResult]]) -> list[SearchResult]:
        """Merge per-strategy result lists into a single ordered list."""
        raise NotImplementedError


class IdentityFusionStrategy(ResultFusionStrategy):
    """Pass-through fusion: concatenate result lists without reordering.

    With a single strategy this returns that strategy's results unchanged, which
    preserves the current single-strategy behavior. It performs no scoring,
    deduplication, or reordering.
    """

    def fuse(self, results: list[list[SearchResult]]) -> list[SearchResult]:
        fused: list[SearchResult] = []
        for result_list in results:
            fused.extend(result_list)
        return fused


class ReciprocalRankFusion(ResultFusionStrategy):
    """Combine ranked result lists using Reciprocal Rank Fusion (RRF).

    Each chunk's fused score is the sum, over every strategy that returned it, of
    ``1 / (k + rank)`` where ``rank`` is the chunk's 1-based position within that
    strategy's list. Chunks are deduplicated by their unique id, so a chunk
    returned by several strategies is merged into a single result whose score
    aggregates all its ranks.

    RRF depends only on rankings, never on the strategies' raw scores, so it is
    fully backend-agnostic: vector similarity, BM25, and any future ranked
    backend combine through the same algorithm without special-casing. ``k``
    dampens the influence of low ranks; the standard default is 60.
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k must be greater than 0")
        self.k = k

    def fuse(self, results: list[list[SearchResult]]) -> list[SearchResult]:
        fused_scores: dict[str, float] = {}
        representatives: dict[str, SearchResult] = {}

        for result_list in results:
            for rank, result in enumerate(result_list, start=1):
                chunk_id = result.chunk.id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                    self.k + rank
                )
                # Keep the first-seen chunk as the canonical representative; the
                # underlying chunk is identical across strategies (same id).
                representatives.setdefault(chunk_id, result)

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )
        return [
            SearchResult(chunk=representatives[chunk_id].chunk, score=fused_scores[chunk_id])
            for chunk_id in ranked_ids
        ]
