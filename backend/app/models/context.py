from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass(frozen=True)
class ContextChunk:
    """A retrieved chunk promoted into the generation context.

    Wraps a :class:`Chunk` with a stable, human-facing ``reference_id`` (``1``,
    ``2``, ...) that the prompt asks the LLM to cite and that the citation mapper
    resolves back to the original chunk. The final reranked ``score`` is carried
    through so downstream layers can order or inspect the context.
    """

    reference_id: int
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class AssembledContext:
    """Deduplicated, budget-bounded context handed to the prompt builder.

    The LLM never sees raw :class:`~app.models.search_result.SearchResult`
    objects; it only ever receives an ``AssembledContext``. Chunks are ordered by
    final reranked relevance and each carries a stable reference id.

    ``notice`` carries an optional retrieval advisory for the prompt — e.g. that
    a requested date filter matched nothing and the sources were fetched by
    semantic relevance instead, so the model must not attribute them to that date.
    """

    chunks: tuple[ContextChunk, ...]
    token_count: int = 0
    notice: str | None = None

    def is_empty(self) -> bool:
        """Return ``True`` when no chunks survived assembly."""
        return not self.chunks
