from __future__ import annotations

from app.models.search_request import SearchRequest
from app.services.bm25_tokenizer import Tokenizer, default_tokenizer


class BM25QueryBuilder:
    """Translate a storage-agnostic ``SearchRequest`` into a BM25 query.

    This is the BM25 analogue of ``QdrantFilterBuilder``: it is the single place
    that knows how to turn a request into the term list the BM25 backend scores
    over. It extracts the searchable free-text portion of the request (the
    semantic query, with metadata filters already stripped out by the analyzer)
    and tokenizes it using the same tokenizer the index used for the corpus, so
    query terms line up with indexed terms.

    Keeping this here (rather than in ``SearchEngine`` or the strategy) means the
    engine stays free of any BM25-specific query construction.
    """

    def __init__(self, tokenizer: Tokenizer = default_tokenizer) -> None:
        self._tokenizer = tokenizer

    def build(self, request: SearchRequest) -> list[str]:
        """Return the tokenized BM25 query for the request."""
        return self._tokenizer(request.semantic_query)
