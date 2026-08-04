from __future__ import annotations

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

    def validate(self, raw_filters: Any) -> list[Filter]:
        """Return the subset of ``raw_filters`` that is valid for the schema."""
        if not isinstance(raw_filters, list):
            return []
        valid: list[Filter] = []
        for raw in raw_filters:
            candidate = self._validate_one(raw)
            if candidate is not None:
                valid.append(candidate)
        return valid

    def _validate_one(self, raw: Any) -> Filter | None:
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

        value = self._coerce_value(field, operator, raw.get("value"))
        if value is None:
            return None

        return Filter(field=field.name, operator=operator, value=value)

    def _coerce_value(
        self, field: MetadataField, operator: Operator, value: Any
    ) -> Any:
        if value is None:
            return None

        if operator is Operator.BETWEEN:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            low = self._coerce_scalar(field, value[0])
            high = self._coerce_scalar(field, value[1])
            if low is None or high is None:
                return None
            return [low, high]

        # A multi-valued equality/contains may arrive as a list of options.
        if isinstance(value, (list, tuple)):
            coerced = [self._coerce_scalar(field, item) for item in value]
            coerced = [item for item in coerced if item is not None]
            return coerced or None

        return self._coerce_scalar(field, value)

    @staticmethod
    def _coerce_scalar(field: MetadataField, value: Any) -> Any:
        """Coerce a scalar to the field's type, or None if it does not fit."""
        if value is None:
            return None

        if field.type is FieldType.BOOLEAN:
            return _coerce_bool(value)
        if field.type is FieldType.NUMBER:
            return _coerce_number(value)
        if field.type is FieldType.DATE:
            return _coerce_date(value)
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


def _coerce_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    try:
        return date.fromisoformat(token[:10]).isoformat()
    except ValueError:
        return None
