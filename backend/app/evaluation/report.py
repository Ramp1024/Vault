from __future__ import annotations

from app.evaluation.metrics import QueryEvaluation
from app.evaluation.runner import EvaluationReport

_SEP = "=" * 90
_SUBSEP = "-" * 90


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
