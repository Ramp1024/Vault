"""Stage 3: the constraint validation pipeline — the system's trust boundary.

The validator is authoritative; the LLM is advisory. Every LLM-proposed
:class:`CandidateConstraint` must clear four gates *in sequence*, and failing any
one removes the constraint:

1. **Schema validation** — can this filter even execute? The field must exist,
   be an enumerable string field (free-text content is a subject, never a
   filter), and the value must snap to a canonical allowed value.
2. **Grounding validation** — is the filter actually supported by the query? A
   schema-valid filter is not enough; there must be positive evidence (an exact
   value match, a known alias, or the model's cited span that really appears in
   the query) plus a cue naming the field.
3. **Necessity / role validation** — is the term a FILTER (narrows the search) or
   the SUBJECT (what is being searched for)? Only FILTER candidates may execute,
   which kills the entire "subject-as-filter" regression class.
4. **Confidence gate** — the final, advisory check. Confidence alone can never
   admit a constraint; it can only veto one that already passed every other gate.

The default is intentionally conservative: no positive evidence means no filter.
A wrong filter can eliminate the correct document entirely, whereas no filter
merely widens the candidate set — so false negatives are preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.filter import Filter, Operator
from app.models.metadata_schema import FieldType, MetadataField, MetadataSchema
from app.processors.constraint_proposal import CandidateConstraint
from app.processors.query_intent import (
    _EXTRA_FIELD_CUES,
    _camel_split,
    _content_tokens,
    _stem,
)

# Grounding score weighting: value evidence dominates (what to filter on) but a
# field cue (that the query names the field) is required for a FILTER role, so
# both contribute. Kept here as the single tunable definition of "grounded".
_VALUE_WEIGHT = 0.6
_FIELD_WEIGHT = 0.4

# Value-evidence strengths by signal, strongest first. Exact and alias matches
# are deterministic (verifiable without trusting the model); the cited-span
# signal is advisory and scaled by the model's confidence.
_EXACT_VALUE_EVIDENCE = 1.0
_ALIAS_VALUE_EVIDENCE = 0.9
_CITED_SPAN_EVIDENCE = 0.7


class ConstraintRole(str, Enum):
    """Whether a term narrows the search (FILTER) or is searched for (SUBJECT)."""

    SUBJECT = "subject"
    FILTER = "filter"


@dataclass(frozen=True)
class SchemaValidationResult:
    """Outcome of stage 3.1: can this constraint be executed at all?"""

    valid: bool
    canonical_value: str | None = None
    operator: Operator | None = None
    reason: str = ""


@dataclass(frozen=True)
class GroundingScore:
    """Interpretable, debuggable grounding evidence for one candidate.

    Each component is in ``[0, 1]``. ``total`` is the weighted combination the
    acceptance gate thresholds on, so a rejection can always be explained by
    pointing at the component that was missing.
    """

    field_evidence: float
    value_evidence: float
    temporal_evidence: float = 0.0

    @property
    def total(self) -> float:
        return round(
            _VALUE_WEIGHT * self.value_evidence + _FIELD_WEIGHT * self.field_evidence,
            4,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "field_evidence": self.field_evidence,
            "value_evidence": self.value_evidence,
            "temporal_evidence": self.temporal_evidence,
            "total": self.total,
        }


@dataclass(frozen=True)
class ConstraintDecision:
    """The full, explainable verdict for a single candidate constraint."""

    candidate: CandidateConstraint
    accepted: bool
    role: ConstraintRole
    grounding: GroundingScore | None
    reason: str
    filter: Filter | None = None


@dataclass(frozen=True)
class ValidationOutcome:
    """Accepted filters plus the per-candidate audit trail behind them."""

    accepted: list[Filter]
    decisions: list[ConstraintDecision]


def _field_cues(field_name: str) -> frozenset[str]:
    """Tokens that count as the query naming ``field_name`` (name + extra cues)."""
    cues = {_stem(part.casefold()) for part in _camel_split(field_name)}
    cues |= _EXTRA_FIELD_CUES.get(field_name, frozenset())
    return frozenset(cues)


class ConstraintValidator:
    """Validate LLM-proposed constraints; only grounded FILTER roles execute.

    Deterministic filters always win: any field already claimed by the
    deterministic analyzer (passed via ``claimed_fields``) rejects competing LLM
    proposals outright, so augmentation can only *add* constraints for fields the
    deterministic stage left open — never override a trusted decision.
    """

    def __init__(
        self,
        schema: MetadataSchema,
        *,
        min_confidence: float = 0.6,
        min_grounding: float = 0.6,
    ) -> None:
        self.schema = schema
        self.min_confidence = min_confidence
        self.min_grounding = min_grounding

    def validate(
        self,
        candidates: list[CandidateConstraint],
        *,
        query: str,
        claimed_fields: frozenset[str] = frozenset(),
    ) -> ValidationOutcome:
        query_tokens = set(_content_tokens(query))
        seen_fields: set[str] = set(claimed_fields)
        accepted: list[Filter] = []
        decisions: list[ConstraintDecision] = []
        for candidate in candidates:
            decision = self._validate_one(candidate, query_tokens, seen_fields)
            decisions.append(decision)
            if decision.accepted and decision.filter is not None:
                accepted.append(decision.filter)
                seen_fields.add(decision.filter.field)
        return ValidationOutcome(accepted=accepted, decisions=decisions)

    def _validate_one(
        self,
        candidate: CandidateConstraint,
        query_tokens: set[str],
        seen_fields: set[str],
    ) -> ConstraintDecision:
        # 3.1 Schema validation — can this filter execute?
        schema_result = self._schema_validate(candidate)
        if not schema_result.valid:
            return _reject(
                candidate, ConstraintRole.FILTER, None, f"schema: {schema_result.reason}"
            )

        # Stage 4 precedence: a field a trusted stage already set is never
        # overridden by an LLM proposal.
        if candidate.field in seen_fields:
            return _reject(
                candidate,
                ConstraintRole.FILTER,
                None,
                "field already constrained (deterministic wins)",
            )

        field = self.schema.get(candidate.field)
        assert field is not None  # guaranteed by schema validation
        canonical = schema_result.canonical_value or candidate.value

        # 3.2 Grounding validation — is the filter supported by the query?
        grounding = self._ground(field, canonical, candidate, query_tokens)
        if grounding.value_evidence <= 0:
            return _reject(
                candidate,
                ConstraintRole.SUBJECT,
                grounding,
                "no value evidence in query",
            )

        # 3.3 Necessity / role — a value not corroborated by a field cue is the
        # search subject, not a constraint.
        if grounding.field_evidence <= 0:
            return _reject(
                candidate,
                ConstraintRole.SUBJECT,
                grounding,
                "no field evidence; value treated as subject",
            )

        if grounding.total < self.min_grounding:
            return _reject(
                candidate,
                ConstraintRole.FILTER,
                grounding,
                f"grounding {grounding.total:.2f} < {self.min_grounding:.2f}",
            )

        # 3.4 Confidence gate — advisory, applied last.
        if candidate.confidence < self.min_confidence:
            return _reject(
                candidate,
                ConstraintRole.FILTER,
                grounding,
                f"confidence {candidate.confidence:.2f} < {self.min_confidence:.2f}",
            )

        filt = Filter(
            field=field.name, operator=schema_result.operator, value=canonical
        )
        return ConstraintDecision(
            candidate=candidate,
            accepted=True,
            role=ConstraintRole.FILTER,
            grounding=grounding,
            reason="accepted",
            filter=filt,
        )

    def _schema_validate(
        self, candidate: CandidateConstraint
    ) -> SchemaValidationResult:
        field = self.schema.get(candidate.field)
        if field is None:
            return SchemaValidationResult(valid=False, reason="unknown field")
        if field.type is not FieldType.STRING or not field.is_enumerable:
            # Free-text / non-enumerable fields hold search subjects, not filters.
            return SchemaValidationResult(valid=False, reason="field not filterable")
        canonical = field.canonical_value(candidate.value)
        if canonical is None:
            return SchemaValidationResult(
                valid=False, reason=f"value '{candidate.value}' not in allowed values"
            )
        operator = Operator.CONTAINS if field.multi else Operator.EQUALS
        return SchemaValidationResult(
            valid=True, canonical_value=canonical, operator=operator
        )

    def _ground(
        self,
        field: MetadataField,
        canonical: str,
        candidate: CandidateConstraint,
        query_tokens: set[str],
    ) -> GroundingScore:
        value_tokens = _content_tokens(canonical)
        exact = bool(value_tokens) and all(t in query_tokens for t in value_tokens)
        alias = self._alias_match(field, canonical, query_tokens)
        evidence_tokens = _content_tokens(candidate.evidence)
        cited = bool(evidence_tokens) and all(
            t in query_tokens for t in evidence_tokens
        )

        if exact:
            value_evidence = _EXACT_VALUE_EVIDENCE
        elif alias:
            value_evidence = _ALIAS_VALUE_EVIDENCE
        elif cited:
            # Advisory: the model mapped a real query span to this value. Trust it
            # only in proportion to its confidence — it is not deterministic.
            value_evidence = round(
                _CITED_SPAN_EVIDENCE * max(0.0, min(1.0, candidate.confidence)), 4
            )
        else:
            value_evidence = 0.0

        field_evidence = 1.0 if (_field_cues(field.name) & query_tokens) else 0.0
        return GroundingScore(
            field_evidence=field_evidence,
            value_evidence=value_evidence,
        )

    @staticmethod
    def _alias_match(
        field: MetadataField, canonical: str, query_tokens: set[str]
    ) -> bool:
        needle = canonical.strip().casefold()
        for value, aliases in field.value_aliases:
            if value.strip().casefold() != needle:
                continue
            for alias in aliases:
                alias_tokens = _content_tokens(alias)
                if alias_tokens and all(t in query_tokens for t in alias_tokens):
                    return True
        return False


def _reject(
    candidate: CandidateConstraint,
    role: ConstraintRole,
    grounding: GroundingScore | None,
    reason: str,
) -> ConstraintDecision:
    return ConstraintDecision(
        candidate=candidate,
        accepted=False,
        role=role,
        grounding=grounding,
        reason=reason,
        filter=None,
    )
