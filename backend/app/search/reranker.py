from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder


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


# Process-wide cache so any given cross-encoder model is loaded at most once,
# regardless of how many CrossEncoderReranker instances exist. Model loading
# (weights + tokenizer) is expensive, so it must never happen per request.
_MODEL_CACHE: dict[str, "CrossEncoder"] = {}


def _load_cross_encoder(model_name: str) -> "CrossEncoder":
    """Return a cached ``CrossEncoder``, loading (and caching) it on first use.

    ``sentence_transformers`` (and its heavy torch dependency) is imported lazily
    here so it is only imported when cross-encoder reranking is actually enabled.
    """
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        _MODEL_CACHE[model_name] = model
    return model


class CrossEncoderReranker(Reranker):
    """Reorder fused candidates with a cross-encoder relevance model.

    The reranker is model-agnostic: it is configured entirely by ``model_name``,
    so swapping to a different cross-encoder is a configuration change requiring
    no code edits. It jointly scores each ``(query, chunk)`` pair, then combines
    that signal with the candidates' incoming order.

    By default it **rank-fuses** the cross-encoder ordering with the incoming
    (fusion/RRF) ordering via reciprocal-rank fusion, so a candidate that was
    already ranked highly is not dumped purely because its text is a weaker
    cross-encoder match. Set ``blend_k=None`` to make the cross-encoder score
    authoritative instead (no blending).

    When a request carries metadata filters, ``skip_when_filtered`` (default
    True) skips reranking entirely: the filter already narrowed results to a
    high-precision set whose RRF order is more reliable than the cross-encoder's
    text-relevance judgement there.

    Candidate generation is bounded: at most ``candidate_pool`` top candidates
    are scored, so the cross-encoder never runs over the entire corpus. Model
    loading is lazy (first ``rerank`` call) and cached process-wide.
    """

    def __init__(
        self,
        model_name: str,
        *,
        candidate_pool: int = 25,
        top_n: int | None = None,
        blend_k: int | None = 60,
        skip_when_filtered: bool = True,
    ) -> None:
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be positive when provided")
        if blend_k is not None and blend_k <= 0:
            raise ValueError("blend_k must be positive when provided")
        self.model_name = model_name
        self.candidate_pool = candidate_pool
        self.top_n = top_n
        self.blend_k = blend_k
        self.skip_when_filtered = skip_when_filtered

    def rerank(
        self, request: SearchRequest, results: list[SearchResult]
    ) -> list[SearchResult]:
        if not results:
            return results

        # A metadata filter already narrowed results to a high-precision set whose
        # fusion (RRF) order is more reliable than the cross-encoder's text
        # relevance judgement — reranking there demotes correct, filter-guaranteed
        # hits. So when filters are present, trust the incoming order and skip.
        if self.skip_when_filtered and request.filters:
            return results

        pool = results[: self.candidate_pool]
        model = _load_cross_encoder(self.model_name)
        pairs = [(request.semantic_query, result.chunk.content) for result in pool]
        scores = [float(score) for score in model.predict(pairs)]

        if self.blend_k is None:
            reranked = self._cross_encoder_order(pool, scores)
        else:
            reranked = self._blended_order(pool, scores, self.blend_k)

        if self.top_n is not None:
            return reranked[: self.top_n]
        return reranked

    @staticmethod
    def _cross_encoder_order(
        pool: list[SearchResult], scores: list[float]
    ) -> list[SearchResult]:
        """Order purely by cross-encoder score (score is authoritative)."""
        reranked = [
            SearchResult(chunk=candidate.chunk, score=score)
            for candidate, score in zip(pool, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked

    @staticmethod
    def _blended_order(
        pool: list[SearchResult], scores: list[float], k: int
    ) -> list[SearchResult]:
        """Reciprocal-rank-fuse the incoming order with the cross-encoder order.

        Each candidate's fused score is ``1/(k+incoming_rank) +
        1/(k+cross_encoder_rank)`` using 0-based ranks. This blends the two
        signals without normalising their (incomparable) raw scores, so the
        prior fusion/filter ordering still counts and can protect a strong
        candidate from an aggressive cross-encoder demotion.
        """
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        ce_rank = {index: rank for rank, index in enumerate(order)}
        fused: list[SearchResult] = []
        for incoming_rank, candidate in enumerate(pool):
            blended = 1.0 / (k + incoming_rank) + 1.0 / (k + ce_rank[incoming_rank])
            fused.append(SearchResult(chunk=candidate.chunk, score=blended))
        fused.sort(key=lambda result: result.score, reverse=True)
        return fused

