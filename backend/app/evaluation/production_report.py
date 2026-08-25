"""Assembled reporting for the production benchmark.

Composes the retrieval, query-analysis, hybrid-contribution, and latency metrics
into a single human-readable report with four sections plus a failure analysis:

1. Overall metrics (Recall@5 / Recall@10 / MRR).
2. Breakdowns by category and by difficulty.
3. Query-analysis and hybrid-contribution metrics.
4. Per-stage latency (average / P50 / P95 / P99).
5. Failure analysis — the top retrieval, analyzer, and reranker failures.

It reuses the plain-text/markdown table renderer from :mod:`app.evaluation.report`
so output matches the rest of the evaluation tooling.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from app.evaluation.analysis_metrics import (
    AnalysisReport,
    AnalyzedCase,
    intent_of_request,
)
from app.evaluation.hybrid_metrics import HybridContribution
from app.evaluation.latency_metrics import StageLatency
from app.evaluation.metrics import QueryEvaluation, RetrievalMetric
from app.evaluation.pipeline import PipelineRun
from app.evaluation.report import _abbrev_metric, _render_table

_SEP = "=" * 90


def _metrics_over(
    evaluations: Sequence[QueryEvaluation], metrics: Sequence[RetrievalMetric]
) -> dict[str, float]:
    return {metric.name: metric.compute(evaluations) for metric in metrics}


def format_overall(run: PipelineRun, *, markdown: bool = False) -> str:
    headers = ["Metric", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in run.report.metrics.items()]
    return _render_table(headers, rows, markdown=markdown)


def _grouped_breakdown(
    run: PipelineRun,
    metrics: Sequence[RetrievalMetric],
    key: str,
    *,
    markdown: bool,
) -> str:
    """Metric table grouped by an evaluation-case attribute (category/difficulty)."""
    grouped: dict[str, list[QueryEvaluation]] = defaultdict(list)
    for evaluation in run.report.evaluations:
        grouped[getattr(evaluation.case, key) or "unlabeled"].append(evaluation)

    metric_names = [m.name for m in metrics]
    headers = [key.capitalize(), "N", *[_abbrev_metric(n) for n in metric_names]]
    rows: list[list[str]] = []
    for name, group in sorted(grouped.items()):
        scores = _metrics_over(group, metrics)
        rows.append(
            [name, str(len(group)), *[f"{scores[m]:.3f}" for m in metric_names]]
        )
    overall = _metrics_over(run.report.evaluations, metrics)
    rows.append(
        [
            "OVERALL",
            str(len(run.report.evaluations)),
            *[f"{overall[m]:.3f}" for m in metric_names],
        ]
    )
    return _render_table(headers, rows, markdown=markdown)


def format_category_breakdown(
    run: PipelineRun, metrics: Sequence[RetrievalMetric], *, markdown: bool = False
) -> str:
    return _grouped_breakdown(run, metrics, "category", markdown=markdown)


def format_difficulty_breakdown(
    run: PipelineRun, metrics: Sequence[RetrievalMetric], *, markdown: bool = False
) -> str:
    return _grouped_breakdown(run, metrics, "difficulty", markdown=markdown)


def format_analysis_metrics(
    report: AnalysisReport, *, markdown: bool = False
) -> str:
    headers = ["Query-Analysis Metric", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in report.as_dict().items()]
    return _render_table(headers, rows, markdown=markdown)


def format_hybrid_contribution(
    contribution: HybridContribution, *, markdown: bool = False
) -> str:
    headers = ["Strategy contribution", "Value"]
    rows = [[name, f"{value:.4f}"] for name, value in contribution.as_dict().items()]
    rows.append(["Vector-only wins (count)", str(contribution.vector_only_count)])
    rows.append(["BM25-only wins (count)", str(contribution.bm25_only_count)])
    rows.append(["Hybrid-only wins (count)", str(contribution.hybrid_only_count)])
    return _render_table(headers, rows, markdown=markdown)


def format_latency(
    latencies: Sequence[StageLatency], *, markdown: bool = False
) -> str:
    headers = ["Stage", "Avg (ms)", "P50", "P95", "P99"]
    rows = [
        [
            latency.stage,
            f"{latency.average:.1f}",
            f"{latency.p50:.1f}",
            f"{latency.p95:.1f}",
            f"{latency.p99:.1f}",
        ]
        for latency in latencies
    ]
    return _render_table(headers, rows, markdown=markdown)


def _filter_str(filters) -> str:
    if not filters:
        return "(none)"
    return ", ".join(
        f"{f.field} {f.operator.value} {f.value!r}" for f in filters
    )


def format_retrieval_failures(
    run: PipelineRun, *, primary_k: int = 5, limit: int = 15
) -> str:
    """List the queries where retrieval failed to surface a relevant doc in top-k.

    Ordered worst-first: never-retrieved cases before those merely ranked past
    ``primary_k``. This is the headline "would a real user find it?" failure set.
    """
    scored = [e for e in run.report.evaluations if e.has_expectation]

    def sort_key(e: QueryEvaluation) -> tuple[int, int]:
        rank = e.first_relevant_rank
        return (0, 0) if rank is None else (1, rank)

    failures = [
        e
        for e in scored
        if e.first_relevant_rank is None or e.first_relevant_rank > primary_k
    ]
    failures.sort(key=sort_key)

    lines = [
        (
            f"Top retrieval failures (no relevant doc in top {primary_k}): "
            f"{len(failures)}"
        )
    ]
    for e in failures[:limit]:
        rank = e.first_relevant_rank
        where = "not retrieved" if rank is None else f"rank {rank}"
        cat = e.case.category or "-"
        lines.append(f"  [{cat}] {e.case.id}: {e.case.query}  ({where})")
    return "\n".join(lines)


def format_analyzer_failures(
    analyzed: Sequence[AnalyzedCase], *, limit: int = 15
) -> str:
    """List filter-expecting cases where query analysis produced the wrong filters."""
    from app.evaluation.analysis_metrics import _filter_set

    failures = [
        item
        for item in analyzed
        if item.expected_filters
        and _filter_set(item.expected_filters) != _filter_set(item.produced_filters)
    ]
    lines = [f"Top analyzer failures (wrong/missing filters): {len(failures)}"]
    for item in failures[:limit]:
        cat = item.case.category or "-"
        lines.append(f"  [{cat}] {item.case.id}: {item.case.query}")
        lines.append(f"      expected: {_filter_str(item.expected_filters)}")
        lines.append(f"      produced: {_filter_str(item.produced_filters)}")

    # Also surface pure intent misclassifications (semantic vs metadata/temporal).
    intent_misses = [
        item
        for item in analyzed
        if item.case.expected_intent is not None
        and intent_of_request(item.request) != item.case.expected_intent
        and not item.expected_filters
    ]
    if intent_misses:
        lines.append(f"Intent misclassifications: {len(intent_misses)}")
        for item in intent_misses[:limit]:
            lines.append(
                f"  {item.case.id}: expected {item.case.expected_intent}, "
                f"got {intent_of_request(item.request)}"
            )
    return "\n".join(lines)


def format_reranker_failures(
    before: PipelineRun, after: PipelineRun, *, limit: int = 15
) -> str:
    """List queries the cross-encoder made worse (correct result moved down/out)."""

    def first_relevant(results, expected: set[str]) -> int | None:
        for rank, result in enumerate(results, start=1):
            if result.chunk.document_id in expected:
                return rank
        return None

    regressions: list[str] = []
    for case_id, after_results in after.results_by_case.items():
        before_results = before.results_by_case.get(case_id, [])
        # Recover the case from the evaluation list for its expectation.
        case = next(
            (e.case for e in after.report.evaluations if e.case.id == case_id), None
        )
        if case is None or not case.expected_documents:
            continue
        expected = set(case.expected_documents)
        before_rank = first_relevant(before_results, expected)
        after_rank = first_relevant(after_results, expected)
        worsened = (before_rank is not None and after_rank is None) or (
            before_rank is not None
            and after_rank is not None
            and after_rank > before_rank
        )
        if worsened:
            after_str = "dropped" if after_rank is None else f"rank {after_rank}"
            regressions.append(
                f"  {case_id}: {before_rank} -> {after_str}  ({case.query})"
            )

    lines = [
        (
            f"Top reranker failures (relevant result pushed down/out): "
            f"{len(regressions)}"
        )
    ]
    lines.extend(regressions[:limit])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-analyzer comparison
#
# The functions below render one column (or column group) per analyzer so the
# report shows the *difference each analyzer makes* — the improvement from
# rule-based to LLM to composite — rather than a single analyzer in isolation.
# ---------------------------------------------------------------------------


def format_grouped_comparison(
    runs: Mapping[str, PipelineRun],
    metrics: Sequence[RetrievalMetric],
    key: str,
    *,
    markdown: bool = False,
) -> str:
    """Per-group metric table with one column group per analyzer.

    Rows are the values of an evaluation-case attribute (``category`` or
    ``difficulty``) plus an OVERALL row; columns are one per (metric, analyzer)
    pair, so a single table shows how every analyzer performs on every group.
    """
    if not runs:
        return "(no runs)"

    analyzer_names = list(runs.keys())
    metric_names = [m.name for m in metrics]

    # Stable group ordering, discovered across all runs.
    groups: list[str] = []
    counts: dict[str, int] = {}
    for run in runs.values():
        for evaluation in run.report.evaluations:
            value = getattr(evaluation.case, key) or "unlabeled"
            if value not in counts:
                groups.append(value)
                counts[value] = 0
    for value in groups:
        counts[value] = sum(
            1
            for evaluation in next(iter(runs.values())).report.evaluations
            if (getattr(evaluation.case, key) or "unlabeled") == value
        )

    # analyzer -> group -> metric -> value
    lookup: dict[str, dict[str, dict[str, float]]] = {}
    for name, run in runs.items():
        grouped: dict[str, list[QueryEvaluation]] = defaultdict(list)
        for evaluation in run.report.evaluations:
            grouped[getattr(evaluation.case, key) or "unlabeled"].append(evaluation)
        lookup[name] = {
            group: _metrics_over(members, metrics)
            for group, members in grouped.items()
        }

    headers = [key.capitalize(), "N"]
    for metric in metric_names:
        for name in analyzer_names:
            headers.append(f"{_abbrev_metric(metric)} {name}")

    rows: list[list[str]] = []
    for group in sorted(groups):
        row = [group, str(counts[group])]
        for metric in metric_names:
            for name in analyzer_names:
                value = lookup[name].get(group, {}).get(metric)
                row.append(f"{value:.3f}" if value is not None else "-")
        rows.append(row)

    total = len(next(iter(runs.values())).report.evaluations)
    overall = ["OVERALL", str(total)]
    for metric in metric_names:
        for name in analyzer_names:
            overall.append(f"{runs[name].report.metrics[metric]:.3f}")
    rows.append(overall)

    return _render_table(headers, rows, markdown=markdown)


def format_analysis_comparison(
    reports: Mapping[str, AnalysisReport], *, markdown: bool = False
) -> str:
    """Query-analysis metrics with one column per analyzer.

    This is where the analyzers diverge most: filter recall, filter-generation
    accuracy, intent classification and date resolution should climb from
    rule-based to LLM to composite.
    """
    if not reports:
        return "(no reports)"

    names = list(reports.keys())
    metric_keys = list(next(iter(reports.values())).as_dict().keys())
    headers = ["Query-Analysis Metric", *names]
    rows: list[list[str]] = []
    for metric_key in metric_keys:
        row = [metric_key]
        row.extend(f"{reports[name].as_dict()[metric_key]:.4f}" for name in names)
        rows.append(row)
    return _render_table(headers, rows, markdown=markdown)


def _stage_by_name(
    latencies: Sequence[StageLatency], name: str
) -> StageLatency | None:
    return next((latency for latency in latencies if latency.stage == name), None)


def format_latency_comparison(
    latencies_by_analyzer: Mapping[str, Sequence[StageLatency]],
    *,
    markdown: bool = False,
) -> str:
    """End-to-end and query-analysis latency, one row per analyzer.

    Makes the cost of richer analysis explicit: the LLM stages add latency that
    the query-analysis and end-to-end columns quantify against the improvement
    in retrieval quality shown in the other tables.
    """
    if not latencies_by_analyzer:
        return "(no latencies)"

    headers = [
        "Analyzer",
        "Query Analysis Avg (ms)",
        "Query Analysis P95",
        "End-to-End Avg (ms)",
        "End-to-End P95",
    ]
    rows: list[list[str]] = []
    for name, latencies in latencies_by_analyzer.items():
        analysis = _stage_by_name(latencies, "Query Analysis")
        end_to_end = (
            _stage_by_name(latencies, "End-to-End")
            or _stage_by_name(latencies, "Retrieval Total")
            or (latencies[-1] if latencies else None)
        )
        rows.append(
            [
                name,
                f"{analysis.average:.1f}" if analysis else "-",
                f"{analysis.p95:.1f}" if analysis else "-",
                f"{end_to_end.average:.1f}" if end_to_end else "-",
                f"{end_to_end.p95:.1f}" if end_to_end else "-",
            ]
        )
    return _render_table(headers, rows, markdown=markdown)


def format_improvement_over_baseline(
    baseline: PipelineRun,
    improved: PipelineRun,
    *,
    primary_k: int = 5,
    limit: int = 20,
) -> str:
    """Show which queries an analyzer upgrade fixes — and which it regresses.

    A query is a *fix* when the baseline analyzer failed to surface a relevant
    document in the top ``primary_k`` but the improved analyzer succeeds, and a
    *regression* in the opposite case. This is the direct answer to "what
    difference does each improvement make?".
    """

    def hit(run: PipelineRun, case_id: str) -> bool:
        evaluation = next(
            (e for e in run.report.evaluations if e.case.id == case_id), None
        )
        if evaluation is None or not evaluation.has_expectation:
            return True  # Not scoreable — exclude from both sets.
        rank = evaluation.first_relevant_rank
        return rank is not None and rank <= primary_k

    cases = {e.case.id: e.case for e in improved.report.evaluations}
    fixes: list[str] = []
    regressions: list[str] = []
    for case_id, case in cases.items():
        base_hit = hit(baseline, case_id)
        new_hit = hit(improved, case_id)
        if new_hit and not base_hit:
            fixes.append(f"  [{case.category or '-'}] {case.id}: {case.query}")
        elif base_hit and not new_hit:
            regressions.append(f"  [{case.category or '-'}] {case.id}: {case.query}")

    lines = [
        (
            f"{baseline.name} -> {improved.name}  "
            f"(fixed {len(fixes)}, regressed {len(regressions)}, top {primary_k})"
        ),
        f"Fixed by {improved.name}:",
    ]
    lines.extend(fixes[:limit] or ["  (none)"])
    lines.append(f"Regressed by {improved.name}:")
    lines.extend(regressions[:limit] or ["  (none)"])
    return "\n".join(lines)


def section(title: str, body: str) -> str:
    return f"{_SEP}\n{title}\n{_SEP}\n{body}\n"
