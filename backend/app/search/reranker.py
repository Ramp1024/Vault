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
    no code edits. It jointly scores each ``(query, chunk)`` pair and uses that
    cross-encoder score as the *authoritative* ordering — cosine similarity,
    BM25, and RRF scores are never blended in. The reranker only reorders (and
    optionally trims) the candidate set it is given.

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
    ) -> None:
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be positive when provided")
        self.model_name = model_name
        self.candidate_pool = candidate_pool
        self.top_n = top_n

    def rerank(
        self, request: SearchRequest, results: list[SearchResult]
    ) -> list[SearchResult]:
        if not results:
            return results

        pool = results[: self.candidate_pool]
        model = _load_cross_encoder(self.model_name)
        pairs = [(request.semantic_query, result.chunk.content) for result in pool]
        scores = model.predict(pairs)

        # Cross-encoder score is authoritative: rebuild results with it as the
        # score and sort descending. No blending with prior (RRF/vector/BM25)
        # scores.
        reranked = [
            SearchResult(chunk=candidate.chunk, score=float(score))
            for candidate, score in zip(pool, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)

        if self.top_n is not None:
            return reranked[: self.top_n]
        return reranked
