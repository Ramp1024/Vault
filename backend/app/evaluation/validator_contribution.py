"""LLM Validator Contribution Report.

Isolates exactly what the augmenting analyzer's *accepted* LLM constraints did to
retrieval, versus the deterministic analyzer alone. Both analyzers run the same
hybrid (vector + BM25 + RRF) pipeline over the production dataset, so any
per-query difference is attributable solely to the constraints the validation
pipeline admitted.

For each query it captures the deterministic result, the augmented result, and
the :class:`ValidationOutcome` (which LLM constraints were accepted and why),
then classifies every LLM-affected query as FIXED / HARMED / NEUTRAL by the
change in Recall@5.

Run (needs live Qdrant + Ollama):
    cd backend && PYTHONPATH=. .venv/bin/python -m app.evaluation.validator_contribution
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import QueryEvaluation
from app.evaluation.constraint_metrics import _reason_bucket
from app.evaluation.pipeline import InstrumentedPipeline
from app.evaluation.run_production import (
    DEFAULT_DATASET_PATH,
    PRIMARY_K,
    RETRIEVAL_DEPTH,
    _discover,
    _load_schema,
)
from app.evaluation.runner import evaluation_from_results
from app.models.search_request import SearchRequest
from app.processors.augmenting_analyzer import AugmentingIntentAnalyzer
from app.processors.constraint_proposal import LLMConstraintProposer
from app.processors.constraint_validation import (
    ConstraintDecision,
    ConstraintValidator,
    ValidationOutcome,
)
from app.processors.query_analyzer import QueryAnalyzer
from app.processors.query_intent import DeterministicIntentAnalyzer
from app.search import (
    ReciprocalRankFusion,
    VectorSearchStrategy,
)
from app.search.strategy import BM25SearchStrategy
from app.services.bm25_index import RankBM25Index
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.llm import build_intent_llm
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

_SEP = "=" * 78


def _recall(ev: QueryEvaluation, k: int) -> float:
    expected = ev.expected_document_set
    if not expected:
        return 0.0
    top = set(ev.retrieved_documents[:k])
    return len(expected & top) / len(expected)


def _mrr(ev: QueryEvaluation) -> float:
    rank = ev.first_relevant_rank
    return 1.0 / rank if rank else 0.0


class _CapturingAugmenting(QueryAnalyzer):
    """Wrap the augmenting analyzer, recording each query's ValidationOutcome."""

    def __init__(self, inner: AugmentingIntentAnalyzer) -> None:
        self._inner = inner
        self.outcomes: dict[str, ValidationOutcome | None] = {}
        self.lexical: dict[str, bool] = {}
        self._cache: dict[str, SearchRequest] = {}

    def analyze(self, query: str) -> SearchRequest:
        if query not in self._cache:
            self.lexical[query] = self._inner.deterministic.is_lexical(query)
            request = self._inner.analyze(query)
            self._cache[query] = request
            self.outcomes[query] = self._inner.last_outcome
        return self._cache[query]


class _Caching(QueryAnalyzer):
    def __init__(self, inner: QueryAnalyzer) -> None:
        self._inner = inner
        self._cache: dict[str, SearchRequest] = {}

    def analyze(self, query: str) -> SearchRequest:
        if query not in self._cache:
            self._cache[query] = self._inner.analyze(query)
        return self._cache[query]


@dataclass
class _CaseDelta:
    query: str
    det_r5: float
    aug_r5: float
    det_r10: float
    aug_r10: float
    det_mrr: float
    aug_mrr: float
    outcome: ValidationOutcome | None
    lexical: bool = False

    @property
    def proposed(self) -> list[ConstraintDecision]:
        return list(self.outcome.decisions) if self.outcome else []

    @property
    def rejected(self) -> list[ConstraintDecision]:
        return [d for d in self.proposed if not d.accepted]

    @property
    def accepted(self) -> list[ConstraintDecision]:
        if self.outcome is None:
            return []
        return [d for d in self.outcome.decisions if d.accepted]

    @property
    def has_llm(self) -> bool:
        return bool(self.accepted)

    @property
    def det_failed(self) -> bool:
        return self.det_r5 < 1.0

    @property
    def classification(self) -> str:
        if not self.has_llm:
            return "NEUTRAL"
        if self.aug_r5 > self.det_r5:
            return "FIXED"
        if self.aug_r5 < self.det_r5:
            return "HARMED"
        # Same R@5 but MRR moved — still a soft signal.
        if self.aug_mrr > self.det_mrr:
            return "FIXED"
        if self.aug_mrr < self.det_mrr:
            return "HARMED"
        return "NEUTRAL"


def _hybrid(analyzer: QueryAnalyzer, vector, bm25) -> InstrumentedPipeline:
    return InstrumentedPipeline(
        analyzer,
        [("Vector", vector), ("BM25", bm25)],
        ReciprocalRankFusion(k=settings.RRF_K),
    )


def _evaluate(
    pipeline: InstrumentedPipeline, dataset: EvaluationDataset
) -> dict[str, QueryEvaluation]:
    out: dict[str, QueryEvaluation] = {}
    for case in dataset.cases:
        results, _ = pipeline.run(case.query)
        out[case.id] = evaluation_from_results(case, results)
    return out


def main(dataset_path=DEFAULT_DATASET_PATH) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)

    qdrant = QdrantService(get_qdrant_client())
    fields, multi = _discover(qdrant)
    values = qdrant.discover_property_values()

    index = RankBM25Index()
    BM25Indexer(qdrant_service=qdrant, index=index).rebuild()

    schema = _load_schema(fields, multi, values)

    vector = VectorSearchStrategy(
        embedding_service=EmbeddingService(), qdrant_service=qdrant
    )
    bm25 = BM25SearchStrategy(index=index)

    det_analyzer = _Caching(
        DeterministicIntentAnalyzer(schema, default_top_k=RETRIEVAL_DEPTH)
    )
    aug = AugmentingIntentAnalyzer(
        deterministic=DeterministicIntentAnalyzer(
            schema, default_top_k=RETRIEVAL_DEPTH
        ),
        proposer=LLMConstraintProposer(build_intent_llm(), schema),
        validator=ConstraintValidator(
            schema,
            min_confidence=settings.INTENT_LLM_MIN_CONFIDENCE,
            min_grounding=settings.INTENT_GROUNDING_THRESHOLD,
        ),
    )
    aug_analyzer = _CapturingAugmenting(aug)

    det_evals = _evaluate(_hybrid(det_analyzer, vector, bm25), dataset)
    aug_evals = _evaluate(_hybrid(aug_analyzer, vector, bm25), dataset)

    deltas: list[_CaseDelta] = []
    for case in dataset.cases:
        de, ae = det_evals[case.id], aug_evals[case.id]
        if not de.has_expectation:
            continue
        deltas.append(
            _CaseDelta(
                query=case.query,
                det_r5=_recall(de, 5),
                aug_r5=_recall(ae, 5),
                det_r10=_recall(de, 10),
                aug_r10=_recall(ae, 10),
                det_mrr=_mrr(de),
                aug_mrr=_mrr(ae),
                outcome=aug_analyzer.outcomes.get(case.query),
                lexical=aug_analyzer.lexical.get(case.query, False),
            )
        )

    _print_report(deltas)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_report(deltas: list[_CaseDelta]) -> None:
    n = len(deltas)
    det_failed = [d for d in deltas if d.det_failed]
    llm_cases = [d for d in deltas if d.has_llm]
    fixed = [d for d in llm_cases if d.classification == "FIXED"]
    harmed = [d for d in llm_cases if d.classification == "HARMED"]

    d_r5 = _mean([d.aug_r5 for d in deltas]) - _mean([d.det_r5 for d in deltas])
    d_r10 = _mean([d.aug_r10 for d in deltas]) - _mean([d.det_r10 for d in deltas])
    d_mrr = _mean([d.aug_mrr for d in deltas]) - _mean([d.det_mrr for d in deltas])

    lexical = [d for d in deltas if d.lexical]
    proposed_cases = [d for d in deltas if d.proposed]
    total_proposals = sum(len(d.proposed) for d in deltas)
    total_accepted = sum(len(d.accepted) for d in deltas)
    total_rejected = sum(len(d.rejected) for d in deltas)
    # Non-lexical queries the proposer ran on but returned nothing.
    proposed_empty = [d for d in deltas if not d.lexical and not d.proposed]
    # Of the deterministic failures, how many did the LLM even try to help?
    failed_with_proposal = [d for d in det_failed if d.proposed]

    reasons: dict[str, int] = {}
    for d in deltas:
        for decision in d.rejected:
            bucket = _reason_bucket(decision.reason)
            reasons[bucket] = reasons.get(bucket, 0) + 1

    print(_SEP)
    print("LLM VALIDATOR CONTRIBUTION REPORT")
    print(_SEP)
    print(f"Total scored queries:                        {n}")
    print(f"Queries where deterministic failed (R@5<1):  {len(det_failed)}")
    print(f"Queries fixed by accepted LLM constraints:   {len(fixed)}")
    print(f"Queries harmed by accepted LLM constraints:  {len(harmed)}")
    print()
    print("Net retrieval gain (Augmenting - Deterministic, over all queries):")
    print(f"  Δ Recall@{PRIMARY_K}:  {d_r5:+.4f}")
    print(f"  Δ Recall@10: {d_r10:+.4f}")
    print(f"  Δ MRR:       {d_mrr:+.4f}")
    print()

    print("Absolute means:")
    print(
        f"  Deterministic  R@5={_mean([d.det_r5 for d in deltas]):.4f}  "
        f"R@10={_mean([d.det_r10 for d in deltas]):.4f}  "
        f"MRR={_mean([d.det_mrr for d in deltas]):.4f}"
    )
    print(
        f"  Augmenting     R@5={_mean([d.aug_r5 for d in deltas]):.4f}  "
        f"R@10={_mean([d.aug_r10 for d in deltas]):.4f}  "
        f"MRR={_mean([d.aug_mrr for d in deltas]):.4f}"
    )
    print()

    print(_SEP)
    print("WHY: PROPOSAL / VALIDATION FUNNEL")
    print(_SEP)
    print(f"Queries lexical-routed (LLM skipped):        {len(lexical)}")
    print(f"Non-lexical queries, proposer returned []:   {len(proposed_empty)}")
    print(f"Queries where LLM proposed >=1 candidate:    {len(proposed_cases)}")
    print(f"Total candidates proposed:                   {total_proposals}")
    print(f"  accepted:                                  {total_accepted}")
    print(f"  rejected:                                  {total_rejected}")
    print(f"Deterministic failures that got a proposal:  "
          f"{len(failed_with_proposal)} / {len(det_failed)}")
    if reasons:
        print("Rejection reasons:")
        for bucket, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {bucket:26s} {count}")
    print()

    # Show what the LLM actually tried but the validator dropped — the most
    # informative signal about whether the gates are too strict or the proposer
    # is off-target.
    if total_rejected:
        print(_SEP)
        print("SAMPLE REJECTED PROPOSALS (up to 15)")
        print(_SEP)
        shown = 0
        for d in deltas:
            for decision in d.rejected:
                cand = decision.candidate
                g = decision.grounding
                ground_str = (
                    f"val={g.value_evidence:.2f},field={g.field_evidence:.2f},"
                    f"total={g.total:.2f}"
                    if g
                    else "n/a"
                )
                print(f'"{d.query}"')
                print(
                    f"    proposed {cand.field}={cand.value!r} conf={cand.confidence:.2f} "
                    f"evidence={cand.evidence!r}"
                )
                print(f"    REJECTED [{decision.role.value}] {decision.reason}  ({ground_str})")
                shown += 1
                if shown >= 15:
                    break
            if shown >= 15:
                break
        print()

    print(_SEP)
    print("TOP 20 ACCEPTED CONSTRAINTS (with reasoning)")
    print(_SEP)
    rows: list[tuple[_CaseDelta, ConstraintDecision]] = []
    for d in llm_cases:
        for decision in d.accepted:
            rows.append((d, decision))

    order = {"FIXED": 0, "HARMED": 1, "NEUTRAL": 2}

    def sort_key(row: tuple[_CaseDelta, ConstraintDecision]):
        d, decision = row
        grounding = decision.grounding.total if decision.grounding else 0.0
        return (order[d.classification], -grounding)

    rows.sort(key=sort_key)

    if not rows:
        print("(no LLM constraints were accepted)")
    for d, decision in rows[:20]:
        g = decision.grounding
        cand = decision.candidate
        filt = decision.filter
        ground_str = (
            f"grounding={g.total:.2f} (val={g.value_evidence:.2f},"
            f"field={g.field_evidence:.2f})"
            if g
            else "grounding=n/a"
        )
        print(f"[{d.classification}] \"{d.query}\"")
        print(
            f"    {filt.field}={filt.value!r}  {ground_str}  conf={cand.confidence:.2f}"
        )
        print(
            f"    evidence={cand.evidence!r}  "
            f"ΔR@5={d.aug_r5 - d.det_r5:+.2f}  ΔMRR={d.aug_mrr - d.det_mrr:+.2f}"
        )
        if cand.rationale:
            print(f"    rationale: {cand.rationale}")
        print()


if __name__ == "__main__":
    main()
