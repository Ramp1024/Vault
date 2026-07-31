"""Lightweight end-to-end generation evaluation.

Runs the full pipeline — retrieval, context building, prompting, generation, and
citation mapping — for each query and records what was retrieved, the generated
answer, its citations, and generation latency. Evaluation here is *qualitative*:
it deliberately performs no automated answer scoring (that belongs to a later
milestone). Its output feeds the markdown diagnostics report for manual review.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.models.answer import GeneratedAnswer
from app.models.context import AssembledContext
from app.services.answer_service import AnswerService


@dataclass(frozen=True)
class GenerationEvaluation:
    """Per-query record of a single end-to-end generation run."""

    query: str
    context: AssembledContext
    answer: GeneratedAnswer
    generation_latency_ms: float
    raw_reference_ids: tuple[int, ...] = ()

    @property
    def retrieved_reference_ids(self) -> tuple[int, ...]:
        """Stable reference ids of every chunk placed in the context."""
        return tuple(item.reference_id for item in self.context.chunks)

    @property
    def validated_reference_ids(self) -> tuple[int, ...]:
        """Reference ids that survived validation into citations."""
        return tuple(citation.reference_id for citation in self.answer.citations)

    @property
    def dropped_reference_ids(self) -> tuple[int, ...]:
        """Raw ids the model emitted that were discarded during validation."""
        validated = set(self.validated_reference_ids)
        return tuple(rid for rid in self.raw_reference_ids if rid not in validated)


def evaluate_query(service: AnswerService, query: str) -> GenerationEvaluation:
    """Run the full pipeline for a single ``query`` and capture the outcome.

    Only the generation (LLM) stage is timed; retrieval latency is covered by the
    dedicated retrieval benchmarks and is not re-measured here.
    """
    normalized_query = query.strip()
    results = service.retrieve(normalized_query)
    context = service.build_context(results)
    prompt = service.build_prompt(normalized_query, context)

    start = time.perf_counter()
    answer = service.generate(prompt, context)
    generation_latency_ms = (time.perf_counter() - start) * 1000.0

    raw_reference_ids = service.citation_mapper.extract_reference_ids(answer.answer)

    return GenerationEvaluation(
        query=normalized_query,
        context=context,
        answer=answer,
        generation_latency_ms=generation_latency_ms,
        raw_reference_ids=raw_reference_ids,
    )


def evaluate_queries(
    service: AnswerService, queries: Iterable[str]
) -> list[GenerationEvaluation]:
    """Run the pipeline for every query, returning one record per query."""
    return [evaluate_query(service, query) for query in queries]


def evaluate_dataset(
    service: AnswerService, cases: Sequence
) -> list[GenerationEvaluation]:
    """Run the pipeline over the ``query`` field of each evaluation case."""
    return [evaluate_query(service, case.query) for case in cases]
