"""Deterministic, schema-aware query intent segmentation.

This module separates a query's **subject** (the thing being searched for, which
feeds BM25/vector retrieval) from its **constraints** (structured metadata
filters). It exists because an LLM analyzer, left to free-form filter generation,
turns the search subject into a filter ("Where did I mention BM25?" ->
``techNotes contains 'BM25'``) and over-constrains retrieval.

The pipeline has three layers, each catching what the others cannot:

1. **Lexical route** — a small set of lookup templates ("where did I mention X",
   "what did I write about X") force a subject-only request: no filter, ever.
2. **Constraint extraction** — for non-lexical queries, candidate filters are
   drawn *only* from a field's known values (``allowed_values``); a value token
   that appears in the query becomes a :class:`MatchedConstraint` carrying the
   evidence that produced it.
3. **Confidence gate** — a tiny additive score (value match, field-name cue)
   decides whether a candidate is emitted, so a value that merely *appears*
   (e.g. "learned about graphs") is not automatically a constraint.

No LLM is involved, so intent segmentation is deterministic, explainable (every
filter carries its evidence), and effectively free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.clock import today as current_date
from app.models.filter import Filter, Operator
from app.models.metadata_schema import FieldType, MetadataSchema
from app.models.search_request import SearchRequest
from app.processors.filter_validator import FilterValidator
from app.processors.query_analyzer import QueryAnalyzer
from app.processors.temporal_field_selector import TemporalFieldSelector
from app.processors.temporal_query import detect_temporal_range

# Lookup phrasings whose subject is the thing being searched for, never a filter.
# Deliberately verb-anchored: possessive/restriction phrasings ("my … notes",
# "which … problems have I") are intentionally excluded so genuine metadata
# queries still reach constraint extraction.
_LEXICAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bwhere did i\b",
        r"\bwhen did i\b",
        r"\bwhich day\b",
        r"\bwhich note",
        r"\bwhat did i\s+(write|wrote|note|noted|say|said|jot|mention|learn|read|discuss)",
        r"\bwhat have i\s+(written|learned|noted|read)",
        r"\bdid i\s+(mention|write|wrote|note|say|discuss|jot|read)\b",
        r"\btell me what i\b",
        r"\bwhat do i\s+(know|remember)\b",
    )
)

# Field-name cue tokens beyond the field's own (camel-split) name. These let a
# short/common value ("Done", "Graphs") count as a constraint only when the query
# also names the field it belongs to.
_EXTRA_FIELD_CUES: dict[str, frozenset[str]] = {
    "status": frozenset({"task", "state"}),
    "leetcodeTopic": frozenset({"problem", "leetcode"}),
}

_STOPWORDS: frozenset[str] = frozenset(
    {"and", "or", "of", "the", "a", "an", "to", "in", "on", "my", "for"}
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.casefold()) if t]


def _stem(token: str) -> str:
    """Very light stemmer: drop a trailing plural 's' on longer tokens."""
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in _tokenize(text) if t not in _STOPWORDS]


def _camel_split(name: str) -> list[str]:
    return [part for part in re.split(r"(?=[A-Z])", name) if part]


@dataclass(frozen=True)
class MatchedConstraint:
    """A candidate metadata filter with the evidence and confidence behind it.

    ``evidence`` is the query fragment (matched value tokens) that produced the
    constraint — it powers both explainable failure reports ("matched token
    'graph'") and subject extraction (the evidence is removed from the subject).
    """

    field: str
    value: str
    confidence: int
    evidence: str

    def as_filter(self) -> Filter:
        return Filter(field=self.field, operator=Operator.EQUALS, value=self.value)


class ConstraintExtractor:
    """Extract candidate metadata constraints from a query's known field values.

    Only enumerable string fields participate: a candidate is proposed when a
    field's known value appears (token-wise, lightly stemmed) in the query. A
    value is "distinctive" enough to stand alone when it has two or more content
    tokens or is a long single token; otherwise it must be corroborated by a
    field-name cue before it can be emitted.
    """

    def __init__(self, schema: MetadataSchema) -> None:
        self._entries: list[tuple[str, str, tuple[str, ...], bool, frozenset[str]]] = []
        for field in schema:
            if field.type is not FieldType.STRING or not field.is_enumerable:
                continue
            cues = self._field_cues(field.name)
            for value in field.allowed_values:
                tokens = tuple(_content_tokens(value))
                if not tokens:
                    continue
                distinctive = len(tokens) >= 2 or len(tokens[0]) >= 7
                self._entries.append((field.name, value, tokens, distinctive, cues))
            # LLM-generated synonyms map a paraphrase to the canonical value. They
            # are inferred rather than literal, so they always require a field cue
            # (distinctive=False) to fire — never on the value word alone.
            for value, aliases in field.value_aliases:
                canonical = field.canonical_value(value) or value
                for alias in aliases:
                    tokens = tuple(_content_tokens(alias))
                    if tokens:
                        self._entries.append(
                            (field.name, canonical, tokens, False, cues)
                        )

    @staticmethod
    def _field_cues(field_name: str) -> frozenset[str]:
        cues = {_stem(part.casefold()) for part in _camel_split(field_name)}
        cues |= _EXTRA_FIELD_CUES.get(field_name, frozenset())
        return frozenset(cues)

    def extract(self, query: str) -> list[MatchedConstraint]:
        query_tokens = set(_content_tokens(query))
        if not query_tokens:
            return []

        matches: list[MatchedConstraint] = []
        for field, value, tokens, distinctive, cues in self._entries:
            if not all(token in query_tokens for token in tokens):
                continue
            has_cue = bool(cues & query_tokens)
            confidence = 1 + (1 if has_cue else 0)
            # A non-distinctive value (e.g. "Done", "Graphs") needs a field cue.
            if not distinctive and not has_cue:
                continue
            matches.append(
                MatchedConstraint(
                    field=field,
                    value=value,
                    confidence=confidence,
                    evidence=" ".join(tokens),
                )
            )
        return matches


class DeterministicIntentAnalyzer(QueryAnalyzer):
    """Schema-aware analyzer that segments a query into subject + constraints.

    Lexical-lookup queries yield a subject-only request. Otherwise, candidate
    constraints are extracted from known field values, validated (snapped to
    canonical casing, invalid/free-text filters dropped), and kept when their
    confidence clears ``min_confidence`` and only the highest-confidence value
    per field survives. The subject is the query with the emitted constraints'
    evidence removed, so the retrieval query never re-encodes a constraint.
    """

    def __init__(
        self,
        schema: MetadataSchema,
        *,
        default_top_k: int = 5,
        min_confidence: int = 1,
        validator: FilterValidator | None = None,
    ) -> None:
        self.schema = schema
        self.default_top_k = default_top_k
        self.min_confidence = min_confidence
        self.extractor = ConstraintExtractor(schema)
        self.validator = validator or FilterValidator(schema)
        # Deterministic temporal field choice (no LLM): prefers the activity date.
        self.temporal_selector = TemporalFieldSelector(schema)

    def analyze(self, query: str) -> SearchRequest:
        normalized = " ".join(query.split()).strip()

        if self._is_lexical(normalized):
            return SearchRequest(
                semantic_query=normalized,
                filters=[],
                top_k=self.default_top_k,
            )

        candidates = self.extractor.extract(normalized)
        constraints = self._select(candidates)
        filters, evidence = self._validate(constraints)
        filters = self._with_temporal(normalized, filters)
        subject = self._subject(normalized, evidence) or normalized
        return SearchRequest(
            semantic_query=subject,
            filters=filters,
            top_k=self.default_top_k,
        )

    def _with_temporal(self, query: str, filters: list[Filter]) -> list[Filter]:
        """Add a deterministic date-range filter on the schema-selected field.

        The date range is resolved deterministically (no LLM); the field is
        chosen by ``TemporalFieldSelector``. If no temporal expression is present
        or no field can be selected, ``filters`` is returned unchanged.
        """
        bounds = detect_temporal_range(query, current_date())
        if bounds is None:
            return filters
        selection = self.temporal_selector.select(query)
        if not selection.selected:
            return filters
        low, high = bounds
        kept = [f for f in filters if f.field != selection.field]
        kept.append(
            Filter(field=selection.field, operator=Operator.BETWEEN, value=[low, high])
        )
        return kept

    @staticmethod
    def _is_lexical(query: str) -> bool:
        lowered = query.casefold()
        return any(pattern.search(lowered) for pattern in _LEXICAL_PATTERNS)

    def _select(self, candidates: list[MatchedConstraint]) -> list[MatchedConstraint]:
        """Keep the best above-threshold candidate per field.

        Ties on confidence are broken toward the more specific value (more matched
        evidence tokens), so "Front end System Design" wins over "System Design"
        when both appear in the query.
        """
        best: dict[str, MatchedConstraint] = {}
        for candidate in candidates:
            if candidate.confidence < self.min_confidence:
                continue
            current = best.get(candidate.field)
            if current is None or self._rank(candidate) > self._rank(current):
                best[candidate.field] = candidate
        return list(best.values())

    @staticmethod
    def _rank(constraint: MatchedConstraint) -> tuple[int, int]:
        return (constraint.confidence, len(constraint.evidence.split()))

    def _validate(
        self, constraints: list[MatchedConstraint]
    ) -> tuple[list[Filter], list[str]]:
        """Validate-first: run candidates through the schema validator, then keep."""
        raw = [
            {"field": c.field, "operator": "=", "value": c.value} for c in constraints
        ]
        valid = self.validator.validate(raw)
        valid_fields = {f.field for f in valid}
        evidence = [c.evidence for c in constraints if c.field in valid_fields]
        return valid, evidence

    @staticmethod
    def _subject(query: str, evidence: list[str]) -> str:
        """Remove the evidence tokens that became filters from the subject."""
        if not evidence:
            return query
        drop: set[str] = set()
        for span in evidence:
            drop.update(_tokenize(span))
        kept = [
            word
            for word in query.split()
            if not _tokenize(word) or _stem(_tokenize(word)[0]) not in drop
        ]
        return " ".join(kept).strip()
