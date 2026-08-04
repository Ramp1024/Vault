from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from app.models.document import Document
from app.models.metadata_schema import (
    FieldType,
    MetadataField,
    MetadataSchema,
    infer_field_type,
    operators_for_type,
)

# The property values are stored under this key in a Document's metadata (and in
# the Qdrant payload), matching the Notion parser's output.
_PROPERTIES_KEY = "properties"


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

    return _build_schema(observed)


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
    return MetadataSchema.from_fields(fields)


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
            )
        )
    return MetadataSchema.from_fields(fields)
