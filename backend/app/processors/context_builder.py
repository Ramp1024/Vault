from __future__ import annotations

from collections.abc import Callable

from app.models.context import AssembledContext, ContextChunk
from app.models.search_result import SearchResult


def _estimate_tokens(text: str) -> int:
    """Cheap, backend-agnostic token estimate (~4 characters per token).

    Deliberately avoids a tokenizer dependency; the budget only needs to be
    approximately right to keep prompts within model limits.
    """
    return max(1, len(text) // 4)


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for overlap comparisons."""
    return " ".join(text.split()).lower()


class ContextBuilder:
    """Assemble reranked search results into an LLM-ready context.

    Responsibilities:
      * Accept the reranked ``SearchResult`` list (highest relevance first).
      * Deduplicate identical or overlapping chunks.
      * Respect a configurable token budget.
      * Preserve document and chunk metadata (the full ``Chunk`` is carried).
      * Order context by final reranked relevance and assign stable reference ids.

    The builder never mutates or inspects retrieval internals; it depends only on
    the public ``SearchResult`` shape, keeping generation independent of the
    retrieval pipeline.
    """

    def __init__(
        self,
        token_budget: int = 2000,
        *,
        deduplicate: bool = True,
        token_estimator: Callable[[str], int] = _estimate_tokens,
    ) -> None:
        self.token_budget = token_budget
        self.deduplicate = deduplicate
        self._estimate_tokens = token_estimator

    def build(self, results: list[SearchResult]) -> AssembledContext:
        """Build an :class:`AssembledContext` from reranked results.

        Args:
            results: Reranked search results ordered by descending relevance.

        Returns:
            An ``AssembledContext`` whose chunks fit within the token budget,
            are free of duplicates/overlaps, and carry stable reference ids
            starting at ``1``.
        """
        selected: list[ContextChunk] = []
        kept_contents: list[str] = []
        used_tokens = 0
        reference_id = 1

        for result in results:
            content = result.chunk.content
            normalized = _normalize(content)

            if self.deduplicate and self._is_duplicate(normalized, kept_contents):
                continue

            chunk_tokens = self._estimate_tokens(content)
            # Always admit the first surviving chunk so the LLM never receives an
            # empty context purely because a single chunk exceeds the budget.
            if selected and used_tokens + chunk_tokens > self.token_budget:
                break

            selected.append(
                ContextChunk(
                    reference_id=reference_id,
                    chunk=result.chunk,
                    score=result.score,
                )
            )
            kept_contents.append(normalized)
            used_tokens += chunk_tokens
            reference_id += 1

        return AssembledContext(chunks=tuple(selected), token_count=used_tokens)

    @staticmethod
    def _is_duplicate(normalized: str, kept_contents: list[str]) -> bool:
        """Return ``True`` when ``normalized`` overlaps an already-kept chunk.

        Treats a chunk as redundant when its normalized text is contained in, or
        contains, a previously kept chunk. This collapses both exact duplicates
        and the window overlap produced by sliding-window chunking.
        """
        for existing in kept_contents:
            if normalized in existing or existing in normalized:
                return True
        return False
