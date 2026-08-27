"""Stage 7: analyzer-level metrics for the augmenting query analyzer.

These metrics score the *constraint validation pipeline* itself — how many LLM
proposals were made, accepted, or rejected, and why — independent of end-to-end
retrieval quality (which the overall benchmark already reports as the Retrieval
Delta between the Deterministic and Augmenting analyzers).

They answer "is the LLM contributing safely?":

* **Constraint Acceptance / Rejection Rate** — of every proposal the LLM made,
  how many the validator admitted vs. dropped.
* **LLM Proposal Success Rate** — of the queries where the LLM proposed anything,
  how many ended up with at least one accepted constraint.
* **Grounding Accuracy** — of accepted constraints (which carry ground-truth from
  ``expected_filters`` when available), how many were actually correct — the
  precision of what the LLM was allowed to add.
* **Rejection Reasons** — a histogram of why proposals were dropped, so the
  dominant failure mode (schema / grounding / necessity / confidence) is visible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.models.filter import Filter
from app.processors.constraint_validation import ConstraintDecision, ValidationOutcome


def _reason_bucket(reason: str) -> str:
    """Coarse-bucket a rejection reason for the histogram."""
    lowered = reason.casefold()
    if lowered.startswith("schema"):
        return "schema"
    if "deterministic wins" in lowered:
        return "deterministic_precedence"
    if "subject" in lowered:
        return "necessity_subject"
    if "grounding" in lowered:
        return "grounding"
    if "confidence" in lowered:
        return "confidence"
    if "value evidence" in lowered:
        return "grounding"
    return "other"


@dataclass(frozen=True)
class ConstraintMetricsReport:
    """Aggregate acceptance/rejection metrics for the validation pipeline."""

    queries: int
    queries_with_proposals: int
    proposals: int
    accepted: int
    rejected: int
    accepted_correct: int
    accepted_scored: int
    queries_with_accept: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposals if self.proposals else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.proposals if self.proposals else 0.0

    @property
    def proposal_success_rate(self) -> float:
        """Fraction of proposing queries that yielded >=1 accepted constraint."""
        return (
            self.queries_with_accept / self.queries_with_proposals
            if self.queries_with_proposals
            else 0.0
        )

    @property
    def grounding_accuracy(self) -> float:
        """Precision of accepted constraints against ground-truth filters."""
        return (
            self.accepted_correct / self.accepted_scored
            if self.accepted_scored
            else 1.0
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "Constraint Acceptance Rate": self.acceptance_rate,
            "Constraint Rejection Rate": self.rejection_rate,
            "LLM Proposal Success Rate": self.proposal_success_rate,
            "Grounding Accuracy": self.grounding_accuracy,
        }


def _accepted_is_correct(
    decision: ConstraintDecision, expected: Sequence[Filter]
) -> bool:
    """True when an accepted constraint matches an expected filter (field+value)."""
    produced = decision.filter
    if produced is None:
        return False
    for exp in expected:
        if exp.field != produced.field:
            continue
        if _same_value(exp.value, produced.value):
            return True
    return False


def _same_value(expected: object, produced: object) -> bool:
    if isinstance(expected, str) and isinstance(produced, str):
        return expected.strip().casefold() == produced.strip().casefold()
    return expected == produced


def build_constraint_metrics(
    outcomes: Sequence[tuple[ValidationOutcome | None, Sequence[Filter]]],
) -> ConstraintMetricsReport:
    """Aggregate per-query ``(ValidationOutcome, expected_filters)`` pairs.

    ``expected_filters`` is used only to score grounding accuracy; pass an empty
    sequence when ground truth is unavailable (those decisions are then excluded
    from the accuracy denominator rather than penalised).
    """
    proposals = accepted = rejected = 0
    accepted_correct = accepted_scored = 0
    queries_with_proposals = queries_with_accept = 0
    reasons: Counter[str] = Counter()

    for outcome, expected in outcomes:
        if outcome is None or not outcome.decisions:
            continue
        queries_with_proposals += 1
        query_had_accept = False
        for decision in outcome.decisions:
            proposals += 1
            if decision.accepted:
                accepted += 1
                query_had_accept = True
                if expected:
                    accepted_scored += 1
                    if _accepted_is_correct(decision, expected):
                        accepted_correct += 1
            else:
                rejected += 1
                reasons[_reason_bucket(decision.reason)] += 1
        if query_had_accept:
            queries_with_accept += 1

    return ConstraintMetricsReport(
        queries=len(outcomes),
        queries_with_proposals=queries_with_proposals,
        proposals=proposals,
        accepted=accepted,
        rejected=rejected,
        accepted_correct=accepted_correct,
        accepted_scored=accepted_scored,
        rejection_reasons=dict(reasons),
        queries_with_accept=queries_with_accept,
    )
