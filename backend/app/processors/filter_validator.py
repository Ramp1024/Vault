from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.models.filter import Filter, Operator
from app.models.metadata_schema import (
    OP_BETWEEN,
    OP_CONTAINS,
    OP_EQUALS,
    OP_GT,
    OP_GTE,
    OP_LT,
    OP_LTE,
    FieldType,
    MetadataField,
    MetadataSchema,
)

# Map the schema's terse operator tokens onto the storage-agnostic domain
# Operator enum. This is the single translation point between the LLM contract
# and the retrieval filter model.
_OPERATOR_TOKENS: dict[str, Operator] = {
    OP_EQUALS: Operator.EQUALS,
    OP_CONTAINS: Operator.CONTAINS,
    OP_GT: Operator.GT,
    OP_LT: Operator.LT,
    OP_GTE: Operator.GTE,
    OP_LTE: Operator.LTE,
    OP_BETWEEN: Operator.BETWEEN,
    # Accept a few natural aliases a model might emit.
    "==": Operator.EQUALS,
    "eq": Operator.EQUALS,
}

# A four-digit year the user typed explicitly (e.g. "July 2023"). When present
# we trust the model's year; when absent, any year it emits is a guess and is
# normalized against the reference date.
_EXPLICIT_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


class FilterValidator:
    """Validate LLM-proposed filters against a :class:`MetadataSchema`.

    Guarantees the retrieval engine never receives an invalid metadata filter:
    every accepted filter references a real field, uses an operator that field
    supports, and carries a value coerced to the field's declared type. Invalid
    candidates are silently dropped rather than raising, so a single bad
    suggestion never fails the whole request.
    """

    def __init__(self, schema: MetadataSchema) -> None:
        self.schema = schema

    def validate(
        self,
        raw_filters: Any,
        *,
        query: str | None = None,
        today: date | None = None,
    ) -> list[Filter]:
        """Return the subset of ``raw_filters`` that is valid for the schema.

        When ``query`` is supplied and contains no explicit four-digit year, any
        year the model attached to a date value is treated as a guess and
        normalized to the most recent occurrence on/before ``today`` (defaulting
        to the current date). This corrects a common local-model failure where a
        year-less phrase like "Jul 27" is resolved to a stale training-era year.
        An explicit year in the user's text is always trusted.
        """
        if not isinstance(raw_filters, list):
            return []
        reference = today or date.today()
        normalize_year = query is not None and not _EXPLICIT_YEAR.search(query)
        valid: list[Filter] = []
        for raw in raw_filters:
            candidate = self._validate_one(raw, reference, normalize_year)
            if candidate is not None:
                valid.append(candidate)
        return valid

    def _validate_one(
        self, raw: Any, reference: date, normalize_year: bool
    ) -> Filter | None:
        if not isinstance(raw, dict):
            return None

        name = raw.get("field")
        if not isinstance(name, str):
            return None
        field = self.schema.get(name)
        if field is None:
            return None

        operator_token = raw.get("operator")
        if not isinstance(operator_token, str):
            return None
        token = operator_token.strip().lower()
        if not field.supports(token):
            return None
        operator = _OPERATOR_TOKENS.get(token)
        if operator is None:
            return None

        value = self._coerce_value(
            field, operator, raw.get("value"), reference, normalize_year
        )
        if value is None:
            return None

        value = self._validate_membership(field, value)
        if value is None:
            return None

        return Filter(field=field.name, operator=operator, value=value)

    @staticmethod
    def _validate_membership(field: MetadataField, value: Any) -> Any:
        """Enforce value-level validity for string fields.

        Enumerable fields (a closed set of known values) accept only values that
        match one of those values, snapped to their canonical casing; anything
        else is dropped. Free-text string fields (e.g. long notes) are never
        filterable — their content is a search *subject*, not a constraint — so
        any proposed filter on them is rejected. Non-string fields are unaffected.
        """
        if field.type is not FieldType.STRING:
            return value

        if not field.is_enumerable:
            # Free-text field: refuse to turn its content into a metadata filter.
            return None

        if isinstance(value, list):
            snapped = [field.canonical_value(str(item)) for item in value]
            snapped = [item for item in snapped if item is not None]
            return snapped or None

        return field.canonical_value(str(value))

    def _coerce_value(
        self,
        field: MetadataField,
        operator: Operator,
        value: Any,
        reference: date,
        normalize_year: bool,
    ) -> Any:
        if value is None:
            return None

        if operator is Operator.BETWEEN:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            low = self._coerce_scalar(field, value[0], reference, normalize_year)
            high = self._coerce_scalar(field, value[1], reference, normalize_year)
            if low is None or high is None:
                return None
            return [low, high]

        # A multi-valued equality/contains may arrive as a list of options.
        if isinstance(value, (list, tuple)):
            coerced = [
                self._coerce_scalar(field, item, reference, normalize_year)
                for item in value
            ]
            coerced = [item for item in coerced if item is not None]
            return coerced or None

        return self._coerce_scalar(field, value, reference, normalize_year)

    @staticmethod
    def _coerce_scalar(
        field: MetadataField, value: Any, reference: date, normalize_year: bool
    ) -> Any:
        """Coerce a scalar to the field's type, or None if it does not fit."""
        if value is None:
            return None

        if field.type is FieldType.BOOLEAN:
            return _coerce_bool(value)
        if field.type is FieldType.NUMBER:
            return _coerce_number(value)
        if field.type is FieldType.DATE:
            return _coerce_date(value, reference, normalize_year)
        # STRING
        text = str(value).strip()
        return text or None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        if "." in text:
            return float(text)
        return int(text)
    except (TypeError, ValueError):
        return None


def _coerce_date(value: Any, reference: date, normalize_year: bool) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    try:
        parsed = date.fromisoformat(token[:10])
    except ValueError:
        return None
    if normalize_year:
        parsed = _clamp_year_to_recent(parsed, reference)
    return parsed.isoformat()


def _clamp_year_to_recent(value: date, reference: date) -> date:
    """Move ``value`` to the most recent occurrence of its month/day on or
    before ``reference``.

    Only invoked when the user did not state a year, so the model's year is a
    guess. Uses the reference year, stepping back one year if that day has not
    happened yet this year.
    """
    candidate = _replace_year(value, reference.year)
    if candidate > reference:
        candidate = _replace_year(value, reference.year - 1)
    return candidate


def _replace_year(value: date, year: int) -> date:
    try:
        return value.replace(year=year)
    except ValueError:
        # Feb 29 on a non-leap year: fall back to Feb 28.
        return value.replace(year=year, day=28)
