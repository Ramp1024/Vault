"""Unit tests for the SearchEngine orchestration, focused on fail-open retrieval.

A metadata filter that matches nothing (e.g. a mis-resolved date or wrong axis)
must not starve generation of context: the engine retries once without filters so
retrieval degrades to semantic search instead of returning an empty set.
"""

from __future__ import annotations

from app.models.chunk import Chunk
from app.models.filter import Filter, Operator
from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.search.engine import SearchEngine


class _FixedAnalyzer:
    """Query analyzer stub that always returns a pre-built request."""

    def __init__(self, request: SearchRequest) -> None:
        self._request = request

    def analyze(self, query: str) -> SearchRequest:
        return self._request


class _FilterAwareStrategy:
    """Returns hits only when the request carries no filters (simulates a filter
    that matches nothing but whose semantic query still has candidates)."""

    def __init__(self, unfiltered_hits: list[SearchResult]) -> None:
        self._unfiltered_hits = unfiltered_hits
        self.filter_calls: list[list[Filter]] = []

    def search(self, request: SearchRequest) -> list[SearchResult]:
        self.filter_calls.append(list(request.filters))
        return [] if request.filters else self._unfiltered_hits


def _result(doc_id: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=f"{doc_id}_0",
            document_id=doc_id,
            document_title=doc_id,
            content=doc_id,
            chunk_index=0,
            metadata={},
        ),
        score=1.0,
    )


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.chunk.document_id for r in results]


def _filtered_request() -> SearchRequest:
    return SearchRequest(
        semantic_query="q",
        filters=[
            Filter(field="date", operator=Operator.BETWEEN, value=["2026-01-01", "2026-01-01"])
        ],
        top_k=5,
    )


def test_fail_open_retries_without_filters_when_filtered_search_is_empty():
    strategy = _FilterAwareStrategy([_result("A"), _result("B")])
    engine = SearchEngine(_FixedAnalyzer(_filtered_request()), [strategy])

    results = engine.search("q")

    assert _ids(results) == ["A", "B"]
    # First attempt carried the filter; the retry dropped it.
    assert strategy.filter_calls[0]  # non-empty filters
    assert strategy.filter_calls[1] == []  # relaxed retry


def test_no_fallback_when_filtered_search_has_hits():
    class _Strategy:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, request: SearchRequest) -> list[SearchResult]:
            self.calls += 1
            return [_result("A")]

    strategy = _Strategy()
    engine = SearchEngine(_FixedAnalyzer(_filtered_request()), [strategy])

    results = engine.search("q")

    assert _ids(results) == ["A"]
    assert strategy.calls == 1  # filtered hits found -> no retry


def test_no_fallback_when_request_has_no_filters():
    strategy = _FilterAwareStrategy([])
    request = SearchRequest(semantic_query="q", filters=[], top_k=5)
    engine = SearchEngine(_FixedAnalyzer(request), [strategy])

    results = engine.search("q")

    assert results == []
    # Only the single attempt runs; there is no filter to relax.
    assert len(strategy.filter_calls) == 1
