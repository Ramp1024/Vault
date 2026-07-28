from __future__ import annotations

from collections.abc import Mapping

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import QueryEvaluation
from app.evaluation.runner import EvaluationReport

_SEP = "=" * 90
_SUBSEP = "-" * 90


def _abbrev_metric(name: str) -> str:
    return name.replace("Recall@", "R@")


def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a left/right-aligned fixed-width text table (first column left)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return "  ".join(out)

    divider = "  ".join("-" * w for w in widths)
    return "\n".join([render(headers), divider, *[render(r) for r in rows]])


def _fmt_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored markdown table (first column left, rest right)."""

    def render(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    alignment = [":---" if i == 0 else "---:" for i in range(len(headers))]
    return "\n".join(
        [render(headers), render(alignment), *[render(r) for r in rows]]
    )


def _render_table(
    headers: list[str], rows: list[list[str]], *, markdown: bool
) -> str:
    return (
        _fmt_markdown_table(headers, rows)
        if markdown
        else _fmt_table(headers, rows)
    )


def format_category_comparison(
    reports: Mapping[str, EvaluationReport], *, markdown: bool = False
) -> str:
    """Render a per-category metric comparison table across named strategies.

    ``reports`` maps a strategy label (e.g. ``"Vector"``) to its evaluation
    report. Rows are query categories (plus an OVERALL row); columns are one per
    (metric, strategy) pair, so a single table shows how each strategy performs
    on each category — the view that reveals where BM25, vector, and future
    hybrid strategies complement one another. Set ``markdown`` for a
    GitHub-flavored markdown table instead of the aligned plain-text one.
    """
    if not reports:
        return "(no reports)"

    strategy_names = list(reports.keys())
    first = next(iter(reports.values()))
    metric_names = list(first.metrics.keys())

    # Categories in a stable, sorted order (category_metrics is already sorted).
    categories: list[str] = []
    counts: dict[str, int] = {}
    for report in reports.values():
        for cat in report.category_metrics:
            if cat.category not in counts:
                categories.append(cat.category)
                counts[cat.category] = cat.count

    # Per-strategy lookup: category -> metric -> value.
    lookup: dict[str, dict[str, dict[str, float]]] = {}
    for label, report in reports.items():
        lookup[label] = {c.category: c.metrics for c in report.category_metrics}

    headers = ["Category", "N"]
    for metric in metric_names:
        for label in strategy_names:
            headers.append(f"{_abbrev_metric(metric)} {label}")

    rows: list[list[str]] = []
    for category in categories:
        row = [category, str(counts[category])]
        for metric in metric_names:
            for label in strategy_names:
                value = lookup[label].get(category, {}).get(metric)
                row.append(f"{value:.3f}" if value is not None else "-")
        rows.append(row)

    total = len(first.evaluations)
    overall = ["OVERALL", str(total)]
    for metric in metric_names:
        for label in strategy_names:
            overall.append(f"{reports[label].metrics[metric]:.3f}")
    rows.append(overall)

    return _render_table(headers, rows, markdown=markdown)


def format_overall_comparison(
    reports: Mapping[str, EvaluationReport],
    *,
    target: str | None = None,
    markdown: bool = False,
) -> str:
    """Render overall metrics per mode plus deltas of one mode against the rest.

    ``target`` (default: the last mode in ``reports``) is the mode whose deltas
    are shown against every other mode, e.g. Hybrid minus Vector and Hybrid minus
    BM25, so the benefit (or cost) of fusion is explicit. Set ``markdown`` for a
    GitHub-flavored markdown table instead of the aligned plain-text one.
    """
    if not reports:
        return "(no reports)"

    mode_names = list(reports.keys())
    target = target or mode_names[-1]
    others = [name for name in mode_names if name != target]
    metric_names = list(next(iter(reports.values())).metrics.keys())

    headers = ["Metric", *mode_names, *[f"\u0394 {target}-{name}" for name in others]]
    rows: list[list[str]] = []
    for metric in metric_names:
        row = [metric]
        row.extend(f"{reports[name].metrics[metric]:.3f}" for name in mode_names)
        for name in others:
            delta = reports[target].metrics[metric] - reports[name].metrics[metric]
            row.append(f"{delta:+.3f}")
        rows.append(row)

    return _render_table(headers, rows, markdown=markdown)


def format_query_diagnostics(
    dataset: EvaluationDataset,
    engines: Mapping[str, object],
    *,
    top_n: int = 3,
) -> str:
    """Render a per-query, per-mode diagnostic report for manual inspection.

    For each case it shows the query, the expected documents/chunks, and the top
    ``top_n`` results from every provided engine (any object exposing
    ``search(query) -> list[SearchResult]``), marking which retrieved chunks
    belong to an expected document. Intended to reveal where each strategy — and
    the hybrid fusion — succeeds or fails.
    """
    lines: list[str] = ["# Retrieval Diagnostics", ""]
    for case in dataset.cases:
        category = f" ({case.category})" if case.category else ""
        lines.append(f"## {case.id}{category}")
        lines.append(f"- **query:** {case.query}")
        lines.append(f"- **expected documents:** {list(case.expected_documents)}")
        lines.append(f"- **expected chunks:** {list(case.expected_chunks)}")
        expected_docs = set(case.expected_documents)

        for name, engine in engines.items():
            results = engine.search(case.query)[:top_n]  # type: ignore[attr-defined]
            lines.append(f"- **{name}** top {top_n}:")
            if not results:
                lines.append("    - (no results)")
                continue
            for rank, result in enumerate(results, start=1):
                marker = "HIT " if result.chunk.document_id in expected_docs else "miss"
                lines.append(
                    f"    {rank}. [{marker}] {result.chunk.document_title} "
                    f":: {result.chunk.id} (score={result.score:.4f})"
                )
        lines.append("")

    return "\n".join(lines)


def format_report(report: EvaluationReport, *, primary_k: int = 5) -> str:
    """Render a human-readable retrieval evaluation report.

    Includes overall metrics, per-query detail, and highlighted problem cases.
    ``primary_k`` defines the pass/fail cutoff (a case passes when a relevant
    document is retrieved within the top ``primary_k``).
    """
    lines: list[str] = [_SEP, "VAULT - RETRIEVAL EVALUATION REPORT", _SEP]

    lines.append(f"Queries: {len(report.evaluations)}")
    lines.append("")
    lines.append("Overall metrics:")
    for name, value in report.metrics.items():
        lines.append(f"  {name:<12} {value:.4f}")
    lines.append(_SUBSEP)

    for evaluation in report.evaluations:
        lines.extend(_format_case(evaluation, primary_k))
        lines.append("")

    lines.append(_SUBSEP)
    lines.extend(_format_highlights(report.evaluations, primary_k))
    lines.append(_SEP)
    return "\n".join(lines)


def _passed(evaluation: QueryEvaluation, primary_k: int) -> bool:
    rank = evaluation.first_relevant_rank
    return rank is not None and rank <= primary_k


def _format_case(evaluation: QueryEvaluation, primary_k: int) -> list[str]:
    case = evaluation.case
    rank = evaluation.first_relevant_rank
    status = "PASS" if _passed(evaluation, primary_k) else "FAIL"
    category = f" ({case.category})" if case.category else ""
    return [
        f"[{status}] {case.id}{category}",
        f"  query:              {case.query}",
        f"  expected documents: {list(case.expected_documents)}",
        f"  expected chunks:    {list(case.expected_chunks)}",
        f"  retrieved documents:{list(evaluation.retrieved_documents)}",
        f"  retrieved chunks:   {list(evaluation.retrieved_chunks)}",
        f"  first relevant rank:{rank if rank is not None else 'none'}",
    ]


def _format_highlights(
    evaluations: tuple[QueryEvaluation, ...], primary_k: int
) -> list[str]:
    scored = [e for e in evaluations if e.has_expectation]

    no_retrieval = [e for e in scored if e.first_relevant_rank is None]
    outside_top_k = [
        e
        for e in scored
        if e.first_relevant_rank is not None and e.first_relevant_rank > primary_k
    ]
    failed = [e for e in scored if not _passed(e, primary_k)]

    lines = ["Highlights:"]
    lines.append(f"  Failed queries (no relevant doc in top {primary_k}): {len(failed)}")
    for evaluation in failed:
        lines.append(f"    - {evaluation.case.id}: {evaluation.case.query}")

    lines.append(
        f"  Relevant but outside top {primary_k} (rank > {primary_k}): {len(outside_top_k)}"
    )
    for evaluation in outside_top_k:
        lines.append(
            f"    - {evaluation.case.id}: rank {evaluation.first_relevant_rank}"
        )

    lines.append(f"  No relevant retrieval at all: {len(no_retrieval)}")
    for evaluation in no_retrieval:
        lines.append(f"    - {evaluation.case.id}: {evaluation.case.query}")

    return lines
