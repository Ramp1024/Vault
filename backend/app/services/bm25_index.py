from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.search_result import SearchResult
from app.services.bm25_tokenizer import Tokenizer, default_tokenizer


class BM25Index(ABC):
    """Backend abstraction for BM25 indexing and retrieval.

    Owns the lifecycle of a lexical index over domain ``Chunk`` objects:
    building it, persisting it, loading it, and executing keyword searches. The
    search strategy depends only on this interface, never on the underlying BM25
    library, so the implementation can be swapped without touching retrieval.

    Search returns backend-agnostic ``SearchResult`` objects, identical to what
    the vector backend returns, so downstream code cannot tell BM25 results from
    vector results.
    """

    @abstractmethod
    def build(self, chunks: Sequence[Chunk]) -> None:
        """Build (or rebuild) the index over the given chunks in memory."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_tokens: Sequence[str], limit: int | None = None
    ) -> list[SearchResult]:
        """Return chunks ranked by BM25 score for the tokenized query.

        Results are ordered by descending score. When ``limit`` is ``None`` all
        positively scored chunks are returned, which lets callers apply metadata
        filtering after retrieval without losing candidates.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path | None = None) -> None:
        """Persist the current index to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path | None = None) -> None:
        """Load a previously persisted index from disk."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str | Path | None = None) -> bool:
        """Return True if a persisted index is available to load."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of chunks currently held by the index."""
        raise NotImplementedError


class RankBM25Index(BM25Index):
    """BM25 index backed by the lightweight, offline ``rank_bm25`` library.

    The index keeps the exact ``Chunk`` objects it was built from (reusing their
    ids, document ids, and metadata) alongside their tokenized form, so it
    references the same logical chunks as the vector store and can reconstruct
    full ``SearchResult`` objects without a separate document model.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        tokenizer: Tokenizer = default_tokenizer,
    ) -> None:
        self._path = Path(path) if path is not None else Path(settings.BM25_INDEX_PATH)
        self._tokenizer = tokenizer
        self._chunks: list[Chunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._corpus_tokens = [self._chunk_tokens(chunk) for chunk in self._chunks]
        # BM25Okapi cannot be constructed from an empty corpus.
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def _chunk_tokens(self, chunk: Chunk) -> list[str]:
        """Tokenize the searchable text of a chunk (title + content)."""
        text = f"{chunk.document_title}\n{chunk.content}"
        return self._tokenizer(text)

    def search(
        self, query_tokens: Sequence[str], limit: int | None = None
    ) -> list[SearchResult]:
        if self._bm25 is None or not self._chunks:
            return []

        tokens = list(query_tokens)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        query_terms = set(tokens)
        # Only documents that share at least one query term are lexical matches.
        # Using term overlap (rather than a score-sign cutoff) keeps results
        # correct even when common terms receive non-positive BM25 IDF on small
        # corpora, while still returning nothing for a query with no shared terms.
        matched = [
            (chunk, float(score))
            for chunk, score, doc_terms in zip(
                self._chunks, scores, self._corpus_tokens
            )
            if not query_terms.isdisjoint(doc_terms)
        ]
        matched.sort(key=lambda pair: pair[1], reverse=True)

        results = [SearchResult(chunk=chunk, score=score) for chunk, score in matched]
        if limit is not None:
            results = results[:limit]
        return results

    def save(self, path: str | Path | None = None) -> None:
        if self._bm25 is None:
            raise RuntimeError("Cannot save an empty BM25 index; build it first")

        target = Path(path) if path is not None else self._path
        target.parent.mkdir(parents=True, exist_ok=True)

        # Persist the chunks and their tokenized corpus rather than the BM25
        # object itself; the scorer is cheaply rebuilt on load and this keeps the
        # artifact independent of the scorer's internal representation.
        payload = {"chunks": self._chunks, "corpus_tokens": self._corpus_tokens}
        with target.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: str | Path | None = None) -> None:
        source = Path(path) if path is not None else self._path
        if not source.exists():
            raise FileNotFoundError(f"No persisted BM25 index at: {source}")

        with source.open("rb") as handle:
            payload = pickle.load(handle)

        self._chunks = list(payload["chunks"])
        self._corpus_tokens = list(payload["corpus_tokens"])
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def exists(self, path: str | Path | None = None) -> bool:
        source = Path(path) if path is not None else self._path
        return source.exists()

    @property
    def size(self) -> int:
        return len(self._chunks)
