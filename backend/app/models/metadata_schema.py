from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Iterable, Mapping


class FieldType(str, Enum):
    """The logical value type of a filterable metadata field.

    The type drives which operators a field supports and how the intent analyzer
    validates candidate values. It is intentionally connector-agnostic: every
    connector maps its native property types onto this small closed set.
    """

    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    NUMBER = "number"


# Operator tokens are the stable contract shared with the LLM (see intent_prompt)
# and validated in the analyzer. They are deliberately terse and human-readable
# so the model can emit them directly.
OP_EQUALS = "="
OP_LT = "<"
OP_GT = ">"
OP_LTE = "<="
OP_GTE = ">="
OP_BETWEEN = "between"
OP_CONTAINS = "contains"


# Temporal roles classify how a date field relates to a note, so temporal
# queries can pick the right axis (a user-assigned content date vs. a system
# activity timestamp) without hardcoding field names.
TEMPORAL_CONTENT = "content_date"
TEMPORAL_ACTIVITY = "activity_date"
TEMPORAL_GENERIC = "generic"
TEMPORAL_ROLES = frozenset({TEMPORAL_CONTENT, TEMPORAL_ACTIVITY, TEMPORAL_GENERIC})


# Default operators derived from a field's type. This is the single place that
# encodes "which operators make sense for which type", so schema discovery never
# has to hardcode operator lists per connector.
_DEFAULT_OPERATORS: dict[FieldType, tuple[str, ...]] = {
    FieldType.DATE: (OP_EQUALS, OP_LT, OP_GT, OP_LTE, OP_GTE, OP_BETWEEN),
    FieldType.NUMBER: (OP_EQUALS, OP_LT, OP_GT, OP_LTE, OP_GTE, OP_BETWEEN),
    FieldType.BOOLEAN: (OP_EQUALS,),
    FieldType.STRING: (OP_EQUALS, OP_CONTAINS),
}


def operators_for_type(field_type: FieldType, *, multi: bool = False) -> tuple[str, ...]:
    """Return the operators supported by a field of ``field_type``.

    A multi-valued string field (e.g. tags) supports ``contains`` membership in
    addition to the scalar defaults.
    """
    operators = _DEFAULT_OPERATORS.get(field_type, (OP_EQUALS,))
    if multi and field_type is FieldType.STRING and OP_CONTAINS not in operators:
        operators = operators + (OP_CONTAINS,)
    return operators


@dataclass(frozen=True)
class MetadataField:
    """A single filterable metadata field discovered during ingestion.

    Attributes:
        name: Canonical (camelCase) field name, matching the payload key used by
            retrieval so validated filters map straight through.
        type: The logical :class:`FieldType`.
        operators: The operators the field supports (a subset of the operator
            tokens above).
        multi: Whether the field holds multiple values (list-like).
        description: Optional human-readable hint surfaced to the LLM.
    """

    name: str
    type: FieldType
    operators: tuple[str, ...]
    multi: bool = False
    description: str | None = None
    allowed_values: tuple[str, ...] = ()
    temporal_role: str | None = None
    # Natural-language synonyms per canonical value, e.g. ("Done", ("completed",
    # "finished")). Generated offline so the deterministic analyzer can match a
    # paraphrased query ("what have I completed") to the real value ("Done")
    # without any query-time LLM call.
    value_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def supports(self, operator: str) -> bool:
        """Return True if ``operator`` is valid for this field."""
        return operator in self.operators

    @property
    def is_enumerable(self) -> bool:
        """True when this field draws from a small, known set of values.

        Only enumerable string fields (e.g. a status or category with a handful
        of legal values) can have candidate filter values validated against a
        closed set. Free-text fields (long notes, tags with unbounded values)
        are not enumerable, so the analyzer must never turn their content into a
        metadata filter — that content is a *search subject*, not a constraint.
        """
        return self.type is FieldType.STRING and bool(self.allowed_values)

    def canonical_value(self, value: str) -> str | None:
        """Return the canonically-cased allowed value matching ``value``, or None.

        Comparison is case- and whitespace-insensitive. Non-enumerable fields
        match nothing, since their values cannot be checked against a known set.
        """
        if not self.is_enumerable:
            return None
        needle = value.strip().casefold()
        for allowed in self.allowed_values:
            if needle == allowed.strip().casefold():
                return allowed
        return None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type.value,
            "operators": list(self.operators),
        }
        if self.multi:
            data["multi"] = True
        if self.description:
            data["description"] = self.description
        if self.allowed_values:
            data["allowed_values"] = list(self.allowed_values)
        if self.temporal_role:
            data["temporal_role"] = self.temporal_role
        if self.value_aliases:
            data["value_aliases"] = {
                value: list(aliases) for value, aliases in self.value_aliases
            }
        return data


@dataclass(frozen=True)
class MetadataSchema:
    """The contract between a connector and the intent analyzer.

    A schema enumerates every filterable metadata field a source exposes. It is
    the *only* thing the LLM intent analyzer is allowed to reason about when
    inferring filters, which keeps the analyzer connector-agnostic: new
    connectors contribute a schema without any prompt or analyzer changes.
    """

    fields: tuple[MetadataField, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return bool(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def get(self, name: str) -> MetadataField | None:
        """Return the field with the given canonical name, or None."""
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        return None

    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    def date_fields(self) -> list[MetadataField]:
        """Return every date-typed field (candidates for temporal filtering)."""
        return [f for f in self.fields if f.type is FieldType.DATE]

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetadataSchema":
        fields_raw = raw.get("fields", []) if isinstance(raw, Mapping) else []
        fields: list[MetadataField] = []
        for item in fields_raw:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            type_value = item.get("type")
            if not isinstance(name, str) or not name:
                continue
            try:
                field_type = FieldType(str(type_value))
            except ValueError:
                field_type = FieldType.STRING
            multi = bool(item.get("multi", False))
            operators_raw = item.get("operators")
            operators = (
                tuple(str(op) for op in operators_raw)
                if isinstance(operators_raw, (list, tuple)) and operators_raw
                else operators_for_type(field_type, multi=multi)
            )
            description = item.get("description")
            allowed_raw = item.get("allowed_values")
            allowed_values = (
                tuple(str(v) for v in allowed_raw)
                if isinstance(allowed_raw, (list, tuple))
                else ()
            )
            role = item.get("temporal_role")
            temporal_role = role if role in TEMPORAL_ROLES else None
            aliases_raw = item.get("value_aliases")
            value_aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
            if isinstance(aliases_raw, Mapping):
                value_aliases = tuple(
                    (str(value), tuple(str(a) for a in aliases))
                    for value, aliases in aliases_raw.items()
                    if isinstance(aliases, (list, tuple))
                )
            fields.append(
                MetadataField(
                    name=name,
                    type=field_type,
                    operators=operators,
                    multi=multi,
                    description=description if isinstance(description, str) else None,
                    allowed_values=allowed_values,
                    temporal_role=temporal_role,
                    value_aliases=value_aliases,
                )
            )
        return cls(fields=tuple(fields))

    @classmethod
    def from_fields(
        cls, fields: Iterable[MetadataField]
    ) -> "MetadataSchema":
        # De-duplicate by name, keeping the first (and merging seen operators is
        # unnecessary here since discovery already unifies types per field).
        seen: dict[str, MetadataField] = {}
        for f in fields:
            if f.name not in seen:
                seen[f.name] = f
        return cls(fields=tuple(seen.values()))


def infer_field_type(name: str, values: Iterable[Any]) -> tuple[FieldType, bool]:
    """Infer a field's :class:`FieldType` and multi-ness from observed values.

    The heuristics are deliberately conservative and connector-agnostic:
    booleans and numbers are detected structurally, lists imply a multi-valued
    string field, and date-shaped strings (or a name containing "date") map to
    the date type. Everything else is a string.
    """
    is_multi = False
    saw_bool = False
    saw_number = False
    saw_date = False
    saw_string = False

    for value in values:
        if isinstance(value, (list, tuple, set)):
            is_multi = True
            for item in value:
                if isinstance(item, str):
                    saw_string = True
            continue
        if isinstance(value, bool):
            saw_bool = True
        elif isinstance(value, (int, float)):
            saw_number = True
        elif isinstance(value, str):
            if _looks_like_date(value):
                saw_date = True
            else:
                saw_string = True

    if is_multi:
        return FieldType.STRING, True
    if saw_string:
        # Any non-date free text present means the field is best treated as text.
        return FieldType.STRING, False
    if saw_date or "date" in name.lower():
        return FieldType.DATE, False
    if saw_bool:
        return FieldType.BOOLEAN, False
    if saw_number:
        return FieldType.NUMBER, False
    return FieldType.STRING, False


def _looks_like_date(value: str) -> bool:
    """Return True if ``value`` is an ISO-8601 date (optionally with a time)."""
    token = value.strip()
    if len(token) < 10:
        return False
    try:
        date.fromisoformat(token[:10])
        return True
    except ValueError:
        return False
