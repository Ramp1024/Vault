"""Hybrid retrieval contribution metrics.

Quantifies *why hybrid retrieval exists* by comparing which queries each
strategy can solve on its own. Running the same dataset through three otherwise
identical pipelines — vector-only, BM25-only, and hybrid fusion — and looking at
the per-query success sets reveals:

* how often each single strategy succeeds, and
* how many queries only the fused hybrid pipeline gets right (its unique lift).

Success is defined as "a relevant document appears in the top ``k`` results",
reusing the same relevance signal as the retrieval metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.metrics import QueryEvaluation


def _succeeds(evaluation: QueryEvaluation, k: int) -> bool:
    rank = evaluation.first_relevant_rank
    return rank is not None and rank <= k


def _success_ids(
    evaluations: Sequence[QueryEvaluation], k: int
) -> set[str]:
    return {
        e.case.id
        for e in evaluations
        if e.has_expectation and _succeeds(e, k)
    }


@dataclass(frozen=True)
class HybridContribution:
    """Per-strategy success rates and hybrid's unique contribution at top-k."""

    k: int
    total: int
    vector_success_rate: float
    bm25_success_rate: float
    hybrid_success_rate: float
    hybrid_only_success_rate: float
    vector_only_count: int
    bm25_only_count: int
    hybrid_only_count: int

    def as_dict(self) -> dict[str, float]:
        return {
            f"Vector success@{self.k}": self.vector_success_rate,
            f"BM25 success@{self.k}": self.bm25_success_rate,
            f"Hybrid success@{self.k}": self.hybrid_success_rate,
            f"Hybrid-only success@{self.k}": self.hybrid_only_success_rate,
        }


def hybrid_contribution(
    vector: Sequence[QueryEvaluation],
    bm25: Sequence[QueryEvaluation],
    hybrid: Sequence[QueryEvaluation],
    *,
    k: int = 5,
) -> HybridContribution:
    """Compute per-strategy success rates and hybrid's unique lift at top-k.

    ``hybrid_only`` counts queries the fused pipeline solves that neither the
    vector-only nor the BM25-only pipeline solve — the clearest measure of what
    fusion buys. All three sequences must cover the same cases (any case with an
    expectation is counted in the denominator).
    """
    scored = [e for e in hybrid if e.has_expectation]
    total = len(scored)

    vector_ok = _success_ids(vector, k)
    bm25_ok = _success_ids(bm25, k)
    hybrid_ok = _success_ids(hybrid, k)

    hybrid_only = hybrid_ok - vector_ok - bm25_ok
    vector_only = vector_ok - bm25_ok - hybrid_ok
    bm25_only = bm25_ok - vector_ok - hybrid_ok

    denom = total or 1
    return HybridContribution(
        k=k,
        total=total,
        vector_success_rate=len(vector_ok) / denom,
        bm25_success_rate=len(bm25_ok) / denom,
        hybrid_success_rate=len(hybrid_ok) / denom,
        hybrid_only_success_rate=len(hybrid_only) / denom,
        vector_only_count=len(vector_only),
        bm25_only_count=len(bm25_only),
        hybrid_only_count=len(hybrid_only),
    )
