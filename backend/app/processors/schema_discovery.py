from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from app.core.config import settings
from app.models.document import Document
from app.models.metadata_schema import (
    TEMPORAL_ACTIVITY,
    TEMPORAL_CONTENT,
    FieldType,
    MetadataField,
    MetadataSchema,
    infer_field_type,
    operators_for_type,
)

# The property values are stored under this key in a Document's metadata (and in
# the Qdrant payload), matching the Notion parser's output.
_PROPERTIES_KEY = "properties"

# Connector system timestamps, stored at the payload top level. They are the
# canonical "activity_date" axis (when a note was created/edited), as opposed to
# a user-assigned "content_date" property.
_SYSTEM_DATE_FIELDS: tuple[str, ...] = ("last_edited_time", "created_time")

# A string field is treated as an enumerable (closed-set) field only when its
# observed values are few and repeat across documents. A handful of distinct
# values is categorical on its own; past that, values must also repeat (low
# distinct-to-total ratio) to count as an enum rather than free text.
_SMALL_ENUM_CARDINALITY = 12
_MAX_ENUM_CARDINALITY = 40
_MAX_ENUM_RATIO = 0.5

# Values that are opaque identifiers (e.g. Notion relation page IDs) or very long
# strings are never human-selectable enum labels, so a field made of them is not
# treated as enumerable even when few distinct values happen to be sampled.
_MAX_ENUM_VALUE_LENGTH = 40
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_label_like(value: str) -> bool:
    return len(value) <= _MAX_ENUM_VALUE_LENGTH and _UUID_RE.match(value) is None


def _string_values(values: Iterable[Any]) -> list[str]:
    """Flatten observed values into the non-empty strings they contain."""
    out: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            out.extend(str(item).strip() for item in value if isinstance(item, str))
        elif isinstance(value, str):
            out.append(value.strip())
    return [s for s in out if s]


def _enum_values(
    field_type: FieldType, values: Iterable[Any]
) -> tuple[str, ...]:
    """Return the closed set of allowed values for a low-cardinality string field.

    Returns an empty tuple for non-string fields, and for string fields whose
    values are too many or too unique to be a categorical enum (i.e. free text).
    """
    if field_type is not FieldType.STRING:
        return ()
    strings = _string_values(values)
    if not strings:
        return ()
    distinct = sorted(set(strings))
    if len(distinct) > _MAX_ENUM_CARDINALITY:
        return ()
    # Opaque IDs / long free text are not categorical labels, even if few.
    if not all(_is_label_like(value) for value in distinct):
        return ()
    # A small absolute number of distinct values is categorical on its own; a
    # larger set must also repeat across documents to be an enum, not free text.
    if len(distinct) <= _SMALL_ENUM_CARDINALITY or len(distinct) <= _MAX_ENUM_RATIO * len(
        strings
    ):
        return tuple(distinct)
    return ()


def schema_from_documents(documents: Iterable[Document]) -> MetadataSchema:
    """Infer a :class:`MetadataSchema` from parsed documents' typed properties.

    This is the connector-agnostic discovery path: any connector that emits
    typed values under ``metadata["properties"]`` automatically contributes a
    schema, so no connector-specific knowledge lives in the LLM prompt or the
    analyzer. Types and operators are derived from the observed values.
    """
    observed: dict[str, list[Any]] = defaultdict(list)
    for document in documents:
        properties = _document_properties(document)
        for name, value in properties.items():
            if value is None:
                continue
            observed[name].append(value)

    return with_temporal_metadata(_build_schema(observed))


def schema_from_property_values(
    observed: Mapping[str, Iterable[Any]]
) -> MetadataSchema:
    """Build a full schema from raw, typed property values per field.

    Unlike :func:`schema_from_indexed_fields`, this infers types from the values
    themselves (so numbers stay numbers), classifies enumerable string fields,
    and assigns temporal roles. Used to regenerate the persisted schema directly
    from indexed payloads without re-contacting the source connector.
    """
    collected: dict[str, list[Any]] = {name: list(vals) for name, vals in observed.items()}
    return with_temporal_metadata(_build_schema(collected))


def schema_from_indexed_fields(
    field_names: Iterable[str],
    multi_fields: Iterable[str] = (),
) -> MetadataSchema:
    """Build a schema from discovered payload field names when values are absent.

    Used as a fallback when no persisted schema exists (e.g. evaluating an
    already-indexed collection). Without sample values, types are inferred from
    the field name alone (``*date*`` -> date, list-valued -> multi string,
    everything else -> string), which is enough for the analyzer to validate
    against real fields.
    """
    multi = {name for name in multi_fields}
    fields: list[MetadataField] = []
    for name in field_names:
        if name in multi:
            field_type, is_multi = FieldType.STRING, True
        elif "date" in name.lower():
            field_type, is_multi = FieldType.DATE, False
        else:
            field_type, is_multi = FieldType.STRING, False
        fields.append(
            MetadataField(
                name=name,
                type=field_type,
                operators=operators_for_type(field_type, multi=is_multi),
                multi=is_multi,
            )
        )
    return with_temporal_metadata(MetadataSchema.from_fields(fields))


def _document_properties(document: Document) -> Mapping[str, Any]:
    properties = document.metadata.get(_PROPERTIES_KEY)
    return properties if isinstance(properties, Mapping) else {}


def _build_schema(observed: Mapping[str, list[Any]]) -> MetadataSchema:
    fields: list[MetadataField] = []
    for name in sorted(observed):
        values = observed[name]
        field_type, is_multi = infer_field_type(name, values)
        fields.append(
            MetadataField(
                name=name,
                type=field_type,
                operators=operators_for_type(field_type, multi=is_multi),
                multi=is_multi,
                allowed_values=_enum_values(field_type, values),
            )
        )
    return MetadataSchema.from_fields(fields)


def enrich_schema_with_values(
    schema: MetadataSchema, values_by_field: Mapping[str, Iterable[Any]]
) -> MetadataSchema:
    """Return a copy of ``schema`` with ``allowed_values`` filled from observed values.

    Used when a schema was built without sample values (e.g. loaded from a
    persisted definition or discovered from indexed field names alone). Each
    string field is classified as enumerable or free-text from the supplied
    values, so the analyzer can validate candidate filter values against a known
    set and refuse to filter on free-text fields. Fields absent from
    ``values_by_field`` are returned unchanged. Temporal roles and the system
    activity-date field are ensured so date fields are query-selectable.
    """
    enriched: list[MetadataField] = []
    for field in schema:
        observed = values_by_field.get(field.name)
        if observed is None:
            enriched.append(field)
            continue
        enriched.append(
            MetadataField(
                name=field.name,
                type=field.type,
                operators=field.operators,
                multi=field.multi,
                description=field.description,
                allowed_values=_enum_values(field.type, observed),
                temporal_role=field.temporal_role,
            )
        )
    return with_temporal_metadata(MetadataSchema.from_fields(enriched))


def _temporal_role_for(name: str) -> str:
    return TEMPORAL_ACTIVITY if name in _SYSTEM_DATE_FIELDS else TEMPORAL_CONTENT


def with_temporal_metadata(schema: MetadataSchema) -> MetadataSchema:
    """Tag date fields with a temporal role and ensure the activity-date field.

    Every date field is classified deterministically — connector system
    timestamps (``last_edited_time``/``created_time``) are ``activity_date`` and
    user-assigned date properties are ``content_date`` — so temporal queries can
    choose an axis by role instead of a hardcoded field name. The configured
    authorship field is injected if the discovered schema lacks it, guaranteeing
    a 100%-coverage fallback axis exists. Existing roles are preserved.
    """
    fields: list[MetadataField] = []
    for field in schema:
        if field.type is FieldType.DATE and field.temporal_role is None:
            fields.append(
                MetadataField(
                    name=field.name,
                    type=field.type,
                    operators=field.operators,
                    multi=field.multi,
                    description=field.description,
                    allowed_values=field.allowed_values,
                    temporal_role=_temporal_role_for(field.name),
                )
            )
        else:
            fields.append(field)

    authorship = settings.AUTHORSHIP_DATE_FIELD
    if authorship and authorship not in {f.name for f in fields}:
        fields.append(
            MetadataField(
                name=authorship,
                type=FieldType.DATE,
                operators=operators_for_type(FieldType.DATE),
                description="When the note was last created or edited (activity time).",
                temporal_role=TEMPORAL_ACTIVITY,
            )
        )
    return MetadataSchema.from_fields(fields)
