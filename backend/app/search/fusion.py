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
