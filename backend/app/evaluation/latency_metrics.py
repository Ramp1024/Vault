"""Per-stage latency statistics (average / P50 / P95 / P99).

The instrumented pipeline records a :class:`~app.evaluation.pipeline.StageTiming`
per query. This module turns a batch of those timings into distribution
statistics for every stage — query analysis, each retrieval strategy, fusion,
reranking, answer generation, and end-to-end — so a benchmark run can report not
just averages but tail latency (P95/P99), which is what actually governs the
felt responsiveness of the assistant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.pipeline import PipelineRun, StageTiming


@dataclass(frozen=True)
class StageLatency:
    """Latency distribution (ms) for a single pipeline stage."""

    stage: str
    average: float
    p50: float
    p95: float
    p99: float

    def as_dict(self) -> dict[str, float]:
        return {
            "avg": self.average,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _stage_latency(stage: str, values: Sequence[float]) -> StageLatency:
    ordered = sorted(values)
    average = sum(ordered) / len(ordered) if ordered else 0.0
    return StageLatency(
        stage=stage,
        average=average,
        p50=_percentile(ordered, 50),
        p95=_percentile(ordered, 95),
        p99=_percentile(ordered, 99),
    )


def _strategy_names(timings: Sequence[StageTiming]) -> list[str]:
    names: list[str] = []
    for timing in timings:
        for name in timing.strategies:
            if name not in names:
                names.append(name)
    return names


def stage_latencies(timings: Sequence[StageTiming]) -> list[StageLatency]:
    """Compute the latency distribution for every stage across ``timings``.

    Stages are ordered as they run: query analysis, each retrieval strategy,
    fusion, reranking, answer generation, then the two roll-ups (retrieval total
    and end-to-end). Answer generation is only included when at least one query
    recorded it, so retrieval-only benchmarks stay uncluttered.
    """
    if not timings:
        return []

    latencies: list[StageLatency] = [
        _stage_latency("Query Analysis", [t.analysis for t in timings])
    ]
    for name in _strategy_names(timings):
        latencies.append(
            _stage_latency(name, [t.strategies.get(name, 0.0) for t in timings])
        )
    latencies.append(_stage_latency("Fusion", [t.fusion for t in timings]))
    latencies.append(_stage_latency("Reranking", [t.rerank for t in timings]))

    if any(t.answer_generation for t in timings):
        latencies.append(
            _stage_latency(
                "Answer Generation", [t.answer_generation for t in timings]
            )
        )
        latencies.append(
            _stage_latency("End-to-End", [t.end_to_end for t in timings])
        )
    else:
        latencies.append(_stage_latency("Retrieval Total", [t.total for t in timings]))

    return latencies


def run_latencies(run: PipelineRun) -> list[StageLatency]:
    """Latency distributions for a single benchmarked pipeline run."""
    return stage_latencies(run.timings)
