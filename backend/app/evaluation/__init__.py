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
from app.evaluation.pipeline import (
    InstrumentedPipeline,
    PipelineRun,
    StageTiming,
    aggregate_timings,
    run_pipeline,
)
from app.evaluation.report import (
    format_category_comparison,
    format_latency_breakdown,
    format_overall_comparison,
    format_performance_table,
    format_query_diagnostics,
    format_rerank_diagnostics,
    format_report,
)
from app.evaluation.runner import (
    CategoryMetrics,
    EvaluationReport,
    RetrievalEvaluator,
    build_report,
    evaluation_from_results,
)

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "QueryEvaluation",
    "RetrievalMetric",
    "RecallAtK",
    "MeanReciprocalRank",
    "default_metrics",
    "RetrievalEvaluator",
    "CategoryMetrics",
    "EvaluationReport",
    "build_report",
    "evaluation_from_results",
    "InstrumentedPipeline",
    "PipelineRun",
    "StageTiming",
    "run_pipeline",
    "aggregate_timings",
    "format_report",
    "format_category_comparison",
    "format_overall_comparison",
    "format_query_diagnostics",
    "format_performance_table",
    "format_latency_breakdown",
    "format_rerank_diagnostics",
]
