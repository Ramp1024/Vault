"""Retrieval evaluation framework.

A backend-agnostic framework for measuring retrieval quality against a golden
dataset. It depends only on the public ``SearchEngine.search(query)`` API and
knows nothing about strategies, Qdrant, rerankers, or query analyzers.

The data model is intentionally designed so it can grow into answer and citation
evaluation later (see ``EvaluationCase``) without a redesign.
"""

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import (
    MeanReciprocalRank,
    QueryEvaluation,
    RecallAtK,
    RetrievalMetric,
    default_metrics,
)
from app.evaluation.report import format_report
from app.evaluation.runner import EvaluationReport, RetrievalEvaluator

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "QueryEvaluation",
    "RetrievalMetric",
    "RecallAtK",
    "MeanReciprocalRank",
    "default_metrics",
    "RetrievalEvaluator",
    "EvaluationReport",
    "format_report",
]
