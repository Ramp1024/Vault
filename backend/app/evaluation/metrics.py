from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.dataset import EvaluationCase


@dataclass(frozen=True)
class QueryEvaluation:
    """The outcome of running one case through retrieval.

    Holds the ranked retrieved identifiers alongside the case, and derives the
    small quantities metrics need. Metrics are computed at *document* granularity
    (stable, unique ``document_id``s); retrieved chunk ids are retained for
    reporting and future chunk-level metrics.
    """

    case: EvaluationCase
    retrieved_documents: tuple[str, ...]
    retrieved_chunks: tuple[str, ...]

    @property
    def has_expectation(self) -> bool:
        return bool(self.case.expected_documents)

    @property
    def expected_document_set(self) -> set[str]:
        return set(self.case.expected_documents)

    def relevant_document_ranks(self) -> list[int]:
        """1-based ranks (in retrieval order) at which an expected document appears."""
        expected = self.expected_document_set
        return [
            rank
            for rank, doc_id in enumerate(self.retrieved_documents, start=1)
            if doc_id in expected
        ]

    @property
    def first_relevant_rank(self) -> int | None:
        ranks = self.relevant_document_ranks()
        return ranks[0] if ranks else None


class RetrievalMetric(ABC):
    """A single retrieval metric computed over a set of query evaluations.

    New metrics (Precision@k, MAP, nDCG, ...) are added by subclassing this
    interface; existing metrics never need to change.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name, e.g. ``Recall@5``."""

    @abstractmethod
    def compute(self, evaluations: Sequence[QueryEvaluation]) -> float:
        """Compute the aggregate metric value across evaluations."""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class RecallAtK(RetrievalMetric):
    """Mean fraction of a case's expected documents found in the top-k results."""

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k

    @property
    def name(self) -> str:
        return f"Recall@{self.k}"

    def compute(self, evaluations: Sequence[QueryEvaluation]) -> float:
        scores: list[float] = []
        for evaluation in evaluations:
            if not evaluation.has_expectation:
                continue
            expected = evaluation.expected_document_set
            top_k = set(evaluation.retrieved_documents[: self.k])
            scores.append(len(expected & top_k) / len(expected))
        return _mean(scores)


class MeanReciprocalRank(RetrievalMetric):
    """Mean of 1/rank of the first relevant document (0 when none retrieved)."""

    @property
    def name(self) -> str:
        return "MRR"

    def compute(self, evaluations: Sequence[QueryEvaluation]) -> float:
        reciprocal_ranks: list[float] = []
        for evaluation in evaluations:
            if not evaluation.has_expectation:
                continue
            rank = evaluation.first_relevant_rank
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        return _mean(reciprocal_ranks)


def default_metrics() -> list[RetrievalMetric]:
    """The metric suite reported by default."""
    return [RecallAtK(5), RecallAtK(10), MeanReciprocalRank()]
