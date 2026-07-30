"""Instrumented retrieval pipeline for benchmarking quality *and* latency.

The production ``SearchEngine`` exposes only ``search(query)`` and its public API
is deliberately frozen. To attribute latency to individual stages (query
analysis, each retrieval strategy, fusion, cross-encoder reranking) without
touching that API, this module mirrors ``SearchEngine.search`` with per-stage
timers. It is a benchmarking harness only — it is never used to serve requests.

Quality is scored through the same helpers the engine-based evaluator uses
(``evaluation_from_results`` / ``build_report``), so both paths compute identical
metrics.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import RetrievalMetric, default_metrics
from app.evaluation.runner import (
    EvaluationReport,
    build_report,
    evaluation_from_results,
)
from app.models.search_result import SearchResult
from app.processors.query_analyzer import QueryAnalyzer
from app.search.fusion import ResultFusionStrategy
from app.search.reranker import NoOpReranker, Reranker
from app.search.strategy import SearchStrategy


@dataclass
class StageTiming:
    """Per-stage wall-clock latency (milliseconds) for a single query."""

    analysis: float = 0.0
    strategies: dict[str, float] = field(default_factory=dict)
    fusion: float = 0.0
    rerank: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class PipelineRun:
    """A benchmarked pipeline: quality report, raw timings, per-case results."""

    name: str
    report: EvaluationReport
    timings: tuple[StageTiming, ...]
    results_by_case: dict[str, list[SearchResult]]

    @property
    def strategy_names(self) -> list[str]:
        names: list[str] = []
        for timing in self.timings:
            for name in timing.strategies:
                if name not in names:
                    names.append(name)
        return names


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class InstrumentedPipeline:
    """Runs the same stages as ``SearchEngine`` while recording per-stage latency.

    Strategies are named so vector vs BM25 retrieval latency is reported
    separately. This mirrors the engine's orchestration exactly; it exists only
    so latency can be measured without changing the frozen ``SearchEngine`` API.
    """

    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        named_strategies: Sequence[tuple[str, SearchStrategy]],
        fusion_strategy: ResultFusionStrategy,
        reranker: Reranker | None = None,
    ) -> None:
        self._analyzer = query_analyzer
        self._named_strategies = list(named_strategies)
        self._fusion = fusion_strategy
        self._reranker = reranker or NoOpReranker()

    def run(self, query: str) -> tuple[list[SearchResult], StageTiming]:
        timing = StageTiming()
        start = time.perf_counter()

        t0 = time.perf_counter()
        request = self._analyzer.analyze(query)
        timing.analysis = _ms(t0)

        per_strategy: list[list[SearchResult]] = []
        for name, strategy in self._named_strategies:
            t = time.perf_counter()
            per_strategy.append(strategy.search(request))
            timing.strategies[name] = _ms(t)

        t = time.perf_counter()
        fused = self._fusion.fuse(per_strategy)
        timing.fusion = _ms(t)

        t = time.perf_counter()
        results = self._reranker.rerank(request, fused)
        timing.rerank = _ms(t)

        timing.total = _ms(start)
        return results, timing


def run_pipeline(
    name: str,
    pipeline: InstrumentedPipeline,
    dataset: EvaluationDataset,
    *,
    metrics: Sequence[RetrievalMetric] | None = None,
) -> PipelineRun:
    """Run a dataset through an instrumented pipeline, collecting quality + latency."""
    metrics = list(metrics) if metrics is not None else default_metrics()
    evaluations = []
    timings: list[StageTiming] = []
    results_by_case: dict[str, list[SearchResult]] = {}

    for case in dataset.cases:
        results, timing = pipeline.run(case.query)
        results_by_case[case.id] = results
        timings.append(timing)
        evaluations.append(evaluation_from_results(case, results))

    report = build_report(tuple(evaluations), metrics)
    return PipelineRun(
        name=name,
        report=report,
        timings=tuple(timings),
        results_by_case=results_by_case,
    )


def aggregate_timings(run: PipelineRun) -> dict[str, float]:
    """Average each stage's latency (ms) across all queries in a run."""
    count = len(run.timings) or 1
    agg: dict[str, float] = {
        "analysis": sum(t.analysis for t in run.timings) / count,
        "fusion": sum(t.fusion for t in run.timings) / count,
        "rerank": sum(t.rerank for t in run.timings) / count,
        "total": sum(t.total for t in run.timings) / count,
    }
    for name in run.strategy_names:
        agg[name] = sum(t.strategies.get(name, 0.0) for t in run.timings) / count
    return agg
