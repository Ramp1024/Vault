"""Markdown diagnostics for generation evaluation.

Renders each :class:`~app.evaluation.generation.GenerationEvaluation` as a
self-contained markdown report so answers can be inspected manually and the
reports double as living documentation of pipeline behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.evaluation.generation import GenerationEvaluation

# Max characters of chunk text shown per source in the retrieved-context section.
_SNIPPET_LIMIT = 400


def _snippet(text: str, limit: int = _SNIPPET_LIMIT) -> str:
    """Collapse whitespace and truncate ``text`` for compact display."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "…"


def format_retrieved_context(evaluation: GenerationEvaluation) -> str:
    """Render the retrieved context as numbered, citable source blocks."""
    if evaluation.context.is_empty():
        return "_No context retrieved._"

    lines: list[str] = []
    for item in evaluation.context.chunks:
        chunk = item.chunk
        lines.append(
            f"[{item.reference_id}] {chunk.document_title} "
            f"(score: {item.score:.4f})"
        )
        lines.append("")
        lines.append(f"> {_snippet(chunk.content)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_citations(evaluation: GenerationEvaluation) -> str:
    """Render resolved citations back to their source documents."""
    citations = evaluation.answer.citations
    if not citations:
        return "_No citations._"

    return "\n".join(
        f"[{citation.reference_id}] {citation.document_title} "
        f"(document: {citation.document_id})"
        for citation in citations
    )


def _format_raw_citations(evaluation: GenerationEvaluation) -> str:
    """Render the raw reference ids the model emitted, before validation."""
    raw = evaluation.raw_reference_ids
    if not raw:
        return "_None._"
    return " ".join(f"[{reference_id}]" for reference_id in raw)


def _format_dropped_citations(evaluation: GenerationEvaluation) -> str:
    """Render reference ids discarded during validation (hallucinated cites)."""
    dropped = evaluation.dropped_reference_ids
    if not dropped:
        return "_None._"
    return " ".join(f"[{reference_id}]" for reference_id in dropped)


def format_query_report(evaluation: GenerationEvaluation) -> str:
    """Render a single query's end-to-end result as a markdown section."""
    answer = evaluation.answer.answer or "_No answer generated._"
    confidence = (
        "n/a"
        if evaluation.answer.confidence is None
        else f"{evaluation.answer.confidence:.2f}"
    )

    return "\n".join(
        [
            "## Query",
            "",
            evaluation.query,
            "",
            "---",
            "",
            "### Retrieved Context",
            "",
            format_retrieved_context(evaluation),
            "",
            "---",
            "",
            "### Generated Answer",
            "",
            answer,
            "",
            "---",
            "",
            "### Raw Citations",
            "",
            _format_raw_citations(evaluation),
            "",
            "---",
            "",
            "### Validated Citations",
            "",
            format_citations(evaluation),
            "",
            "---",
            "",
            "### Dropped",
            "",
            _format_dropped_citations(evaluation),
            "",
            "---",
            "",
            "### Diagnostics",
            "",
            f"- Generation latency: {evaluation.generation_latency_ms:.1f} ms",
            f"- Retrieved chunks: {len(evaluation.context.chunks)}",
            f"- Context tokens (est.): {evaluation.context.token_count}",
            f"- Confidence: {confidence}",
        ]
    )


def format_report(evaluations: Sequence[GenerationEvaluation]) -> str:
    """Render a full markdown report covering every evaluated query."""
    header = [
        "# Generation Evaluation Report",
        "",
        f"Queries evaluated: {len(evaluations)}",
        "",
    ]
    sections = [format_query_report(evaluation) for evaluation in evaluations]
    return "\n".join(header) + "\n\n".join(sections) + "\n"
