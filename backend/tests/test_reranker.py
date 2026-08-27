"""Unit tests for cross-encoder rerank blending.

The blending path is pure (no model), so it is tested directly: a candidate
ranked highly by the incoming fusion order must not be dumped purely because the
cross-encoder scores its text lower — this is what protects filter-guaranteed
metadata hits from being demoted.
"""

from __future__ import annotations

from app.models.chunk import Chunk
from app.models.filter import Filter, Operator
from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.search.reranker import CrossEncoderReranker


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
        score=0.0,
    )


def _ids(results: list[SearchResult]) -> list[str]:
    return [r.chunk.document_id for r in results]


def test_pure_cross_encoder_order_is_authoritative():
    pool = [_result("A"), _result("B"), _result("C")]  # incoming order
    scores = [0.1, 0.2, 0.9]  # cross-encoder favours C, then B, then A
    order = CrossEncoderReranker._cross_encoder_order(pool, scores)
    assert _ids(order) == ["C", "B", "A"]  # A (incoming #1) dumped to last


def test_blending_protects_the_top_incoming_candidate():
    pool = [_result("A"), _result("B"), _result("C")]  # A is incoming #1
    scores = [0.1, 0.2, 0.9]  # cross-encoder would rank A last
    blended = CrossEncoderReranker._blended_order(pool, scores, k=10)
    # A keeps the top slot despite the weak cross-encoder score, because its
    # strong incoming rank is fused in rather than overridden.
    assert _ids(blended)[0] == "A"
    assert "A" in _ids(blended)[:2]


def test_blend_disabled_matches_pure_cross_encoder(monkeypatch):
    # blend_k=None should route through the authoritative cross-encoder ordering.
    reranker = CrossEncoderReranker("dummy", blend_k=None)
    assert reranker.blend_k is None


def test_filtered_request_skips_reranking_untouched():
    # A metadata filter guarantees relevance; the reranker trusts fusion order and
    # does not load the model or reorder (returns the exact same list).
    pool = [_result("A"), _result("B"), _result("C")]
    request = SearchRequest(
        semantic_query="q",
        filters=[Filter(field="status", operator=Operator.EQUALS, value="Done")],
    )
    reranker = CrossEncoderReranker("dummy")  # model would fail to load if called
    out = reranker.rerank(request, pool)
    assert out is pool  # returned unchanged, no model invoked


def test_unfiltered_request_is_not_skipped():
    reranker = CrossEncoderReranker("dummy")
    assert reranker.skip_when_filtered is True
    request = SearchRequest(semantic_query="q", filters=[])
    assert not (reranker.skip_when_filtered and request.filters)
