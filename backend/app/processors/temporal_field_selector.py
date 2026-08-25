"""Schema-driven selection of which date field a temporal query should filter.

A note can carry several date axes — a user-assigned *content date* and one or
more system *activity dates* (created/last-edited). This selector decides which
axis a temporal query means, using only the schema's declared date fields and
their ``temporal_role``. It is intentionally the *only* temporal decision the LLM
may touch, and even then the LLM may only pick a field **name** from the provided
list — it never computes dates, ranges, or timestamps.

Fallback is fail-safe, never fail-wrong:

* one candidate            -> use it (deterministic)
* several candidates       -> LLM picks a name (if available and confident),
                              else the deterministic default (prefer activity)
* no confident selection   -> no temporal filter at all (caller drops it)

Crucially it never hardcodes a field name like ``last_edited_time`` as a
fallback: a wrong axis can silently return plausible-but-wrong results, so when
nothing can be chosen confidently the temporal filter is dropped and retrieval
falls back to plain semantic search.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.models.metadata_schema import (
    TEMPORAL_ACTIVITY,
    TEMPORAL_CONTENT,
    TEMPORAL_GENERIC,
    MetadataField,
    MetadataSchema,
)
from app.services.llm import LLM

logger = logging.getLogger(__name__)

# Preference order when falling back deterministically: activity dates have full
# coverage and are the safest default for "what did I work on" style queries.
_ROLE_PRIORITY = (TEMPORAL_ACTIVITY, TEMPORAL_CONTENT, TEMPORAL_GENERIC)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class TemporalSelection:
    """The chosen date field for a temporal query (or None to drop the filter)."""

    field: str | None
    confidence: float
    reason: str

    @property
    def selected(self) -> bool:
        return self.field is not None


class TemporalFieldSelector:
    """Pick the date field a temporal query refers to, schema-driven and safe.

    ``llm`` is optional: when omitted (or when only one candidate exists) the
    selection is fully deterministic and free. The LLM, when provided, is asked
    only to choose the most relevant field name among the schema's date fields.
    """

    def __init__(
        self,
        schema: MetadataSchema,
        *,
        llm: LLM | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self.schema = schema
        self.llm = llm
        self.min_confidence = min_confidence

    def _candidates(self) -> list[MetadataField]:
        return [
            f
            for f in self.schema.date_fields()
            if f.temporal_role in {TEMPORAL_CONTENT, TEMPORAL_ACTIVITY, TEMPORAL_GENERIC}
        ]

    def _default(self, candidates: list[MetadataField]) -> MetadataField | None:
        by_name = {f.name: f for f in candidates}
        for role in _ROLE_PRIORITY:
            for field in candidates:
                if field.temporal_role == role:
                    return by_name[field.name]
        return candidates[0] if candidates else None

    def select(self, query: str) -> TemporalSelection:
        candidates = self._candidates()
        if not candidates:
            return TemporalSelection(None, 0.0, "no date fields in schema")
        if len(candidates) == 1:
            return TemporalSelection(candidates[0].name, 1.0, "only date field")

        if self.llm is not None:
            chosen = self._llm_select(query, candidates)
            if chosen is not None:
                field, confidence = chosen
                if field in {c.name for c in candidates} and confidence >= self.min_confidence:
                    return TemporalSelection(field, confidence, "llm selection")
                logger.warning(
                    "Temporal field selection rejected (field=%r conf=%.2f); "
                    "using deterministic default",
                    field,
                    confidence,
                )

        default = self._default(candidates)
        if default is not None:
            return TemporalSelection(
                default.name, 0.6, f"deterministic default ({default.temporal_role})"
            )
        # Never fabricate an axis: drop the temporal filter and fall back to search.
        return TemporalSelection(None, 0.0, "no confident temporal field")

    def _llm_select(
        self, query: str, candidates: list[MetadataField]
    ) -> tuple[str, float] | None:
        prompt = self._build_prompt(query, candidates)
        try:
            raw = self.llm.generate(prompt)
        except Exception:  # pragma: no cover - backend-dependent
            logger.warning("Temporal field selection LLM call failed", exc_info=True)
            return None
        match = _JSON_OBJECT.search(raw or "")
        if match is None:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        field = data.get("field")
        if not isinstance(field, str):
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return field.strip(), confidence

    @staticmethod
    def _build_prompt(query: str, candidates: list[MetadataField]) -> str:
        lines = [
            "Choose which date field a user's question refers to.",
            "You may ONLY pick a field name from the list. Do NOT compute dates,",
            "ranges, or timestamps, and never invent a field name.",
            "",
            "Date fields:",
        ]
        for field in candidates:
            desc = field.description or ""
            role = field.temporal_role or "generic"
            lines.append(f"- {field.name} (role: {role}) {desc}".rstrip())
        lines += [
            "",
            f"Question: {query}",
            "",
            'Respond as JSON: {"field": "<name>", "confidence": <0..1>}',
        ]
        return "\n".join(lines)
