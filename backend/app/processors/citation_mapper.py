from __future__ import annotations

import re

from app.models.answer import Citation
from app.models.context import AssembledContext

# Matches one or more bracketed reference ids, e.g. "[1]", "[2][3]", "[1, 2]".
_REFERENCE_PATTERN = re.compile(r"\[([0-9]+(?:\s*[,;]\s*[0-9]+)*)\]")


class CitationMapper:
    """Resolve bracketed reference ids in an answer to grounded citations.

    The LLM only emits reference numbers (``[1]``); this mapper deterministically
    resolves each id back to the originating chunk in the :class:`AssembledContext`.
    Ids not present in the context are ignored, so citations can never point at
    content that was not retrieved, and the LLM cannot invent sources.
    """

    def map(self, answer: str, context: AssembledContext) -> tuple[Citation, ...]:
        """Extract and resolve citations from ``answer``.

        Args:
            answer: Raw answer text emitted by the LLM, containing ``[n]`` markers.
            context: The context whose reference ids the markers point at.

        Returns:
            Citations in order of first appearance, de-duplicated by reference id,
            containing only ids that exist in ``context``.
        """
        by_reference = {item.reference_id: item for item in context.chunks}

        citations: list[Citation] = []
        seen: set[int] = set()
        for reference_id in self._iter_reference_ids(answer):
            if reference_id in seen:
                continue
            item = by_reference.get(reference_id)
            if item is None:
                continue
            seen.add(reference_id)
            chunk = item.chunk
            citations.append(
                Citation(
                    reference_id=reference_id,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                )
            )

        return tuple(citations)

    def extract_reference_ids(self, answer: str) -> tuple[int, ...]:
        """Return every reference id the model emitted, de-duplicated, in order.

        Unlike :meth:`map`, this performs no validation against a context — it is
        the *raw* set of ids parsed from the model output. Used by diagnostics to
        contrast raw citations against the validated ones (and reveal drops).
        """
        seen: set[int] = set()
        ordered: list[int] = []
        for reference_id in self._iter_reference_ids(answer):
            if reference_id not in seen:
                seen.add(reference_id)
                ordered.append(reference_id)
        return tuple(ordered)

    @staticmethod
    def _iter_reference_ids(answer: str):
        """Yield reference ids in order of appearance within ``answer``."""
        for match in _REFERENCE_PATTERN.finditer(answer):
            for token in re.split(r"[,;]", match.group(1)):
                token = token.strip()
                if token:
                    yield int(token)
