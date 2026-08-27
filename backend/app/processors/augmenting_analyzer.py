"""The augmenting query analyzer: deterministic truth + validated LLM proposals.

This is the top-level analyzer that ties the redesign together. It guarantees
the LLM can only ever *improve* retrieval, never arbitrarily narrow it:

* **Stage 1 (authoritative).** The deterministic analyzer segments the query into
  a subject and trusted metadata/temporal filters. Its output is the baseline.
* **Lexical short-circuit.** If the deterministic stage routed the query as a
  lexical lookup (subject-only), no LLM filter is ever added — the subject stays
  the sole signal.
* **Stage 2 (advisory).** The LLM proposes candidate constraints for the
  remaining fields; it never emits executable filters.
* **Stage 3 (trust boundary).** Every candidate is validated (schema → grounding
  → necessity → confidence). Only grounded FILTER-role candidates survive.
* **Stage 4 (merge).** Deterministic filters always win; validated LLM filters
  are added only for fields the deterministic stage left unclaimed.
* **Stage 5 (subject preservation).** The subject is the deterministic subject,
  verbatim. The LLM never rewrites or removes user terms, so distinctive lexical
  terms always survive.

Any failure in the LLM path degrades to the deterministic result unchanged, so
composition can never make retrieval less reliable than the deterministic
baseline.
"""

from __future__ import annotations

from app.models.search_request import SearchRequest
from app.processors.constraint_proposal import LLMConstraintProposer
from app.processors.constraint_validation import (
    ConstraintValidator,
    ValidationOutcome,
)
from app.processors.query_analyzer import QueryAnalyzer
from app.processors.query_intent import DeterministicIntentAnalyzer


class AugmentingIntentAnalyzer(QueryAnalyzer):
    """Deterministic analysis augmented by validated, grounded LLM proposals.

    The deterministic analyzer is the source of truth for both the subject and
    any filter it produces. The LLM contributes only additional constraints that
    survive the full validation pipeline, and only for fields the deterministic
    stage did not already claim.
    """

    def __init__(
        self,
        deterministic: DeterministicIntentAnalyzer,
        proposer: LLMConstraintProposer,
        validator: ConstraintValidator,
    ) -> None:
        self.deterministic = deterministic
        self.proposer = proposer
        self.validator = validator
        # Last validation audit, exposed for evaluation/telemetry. None until the
        # LLM path runs (e.g. after a lexical short-circuit).
        self.last_outcome: ValidationOutcome | None = None

    def analyze(self, query: str) -> SearchRequest:
        self.last_outcome = None
        base = self.deterministic.analyze(query)

        # Lexical lookups are subject-only by design; never augment them.
        if self.deterministic.is_lexical(query):
            return base

        try:
            candidates = self.proposer.propose(query)
        except Exception:  # pragma: no cover - defensive around the LLM stage
            return base
        if not candidates:
            return base

        claimed = frozenset(f.field for f in base.filters)
        outcome = self.validator.validate(
            candidates, query=query, claimed_fields=claimed
        )
        self.last_outcome = outcome
        if not outcome.accepted:
            return base

        return SearchRequest(
            semantic_query=base.semantic_query,
            filters=list(base.filters) + outcome.accepted,
            top_k=base.top_k,
        )
