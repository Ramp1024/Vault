from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import QueryEvaluation, RetrievalMetric, default_metrics


class SupportsSearch:
    """Structural note: the evaluator only needs an object exposing

        search(query: str) -> list[SearchResult]

    which is the public ``SearchEngine`` contract. It is never given, and never
    inspects, any retrieval implementation detail (strategies, Qdrant, fusion,
    rerankers, analyzers).
    """


@dataclass(frozen=True)
class EvaluationReport:
    """The result of an evaluation run: per-query outcomes plus aggregate metrics."""

    evaluations: tuple[QueryEvaluation, ...]
    metrics: dict[str, float]


class RetrievalEvaluator:
    """Runs a golden dataset through a search engine and computes metrics.

    Depends only on ``search_engine.search(query)`` returning ``SearchResult``
    objects (public domain model). It is completely independent of how retrieval
    is implemented, so it remains unchanged as BM25, hybrid retrieval, fusion,
    rerankers, or LLM query analysis are added.
    """

    def __init__(
        self,
        search_engine,
        metrics: Sequence[RetrievalMetric] | None = None,
    ) -> None:
        self._search_engine = search_engine
        self.metrics = list(metrics) if metrics is not None else default_metrics()

    def evaluate(self, dataset: EvaluationDataset) -> EvaluationReport:
        evaluations = tuple(self._evaluate_case(case) for case in dataset.cases)
        metric_scores = {
            metric.name: metric.compute(evaluations) for metric in self.metrics
        }
        return EvaluationReport(evaluations=evaluations, metrics=metric_scores)

    def _evaluate_case(self, case) -> QueryEvaluation:
        results = self._search_engine.search(case.query)
        retrieved_documents = _dedupe(result.chunk.document_id for result in results)
        retrieved_chunks = tuple(result.chunk.id for result in results)
        return QueryEvaluation(
            case=case,
            retrieved_documents=retrieved_documents,
            retrieved_chunks=retrieved_chunks,
        )


def _dedupe(values) -> tuple[str, ...]:
    """Preserve order while removing duplicates (ranked documents from chunks)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)
