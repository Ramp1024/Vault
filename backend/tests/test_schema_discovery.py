"""Unit tests for enumerable-field classification during schema discovery.

Verify that low-cardinality string fields become enumerable (closed-set) while
free-text fields do not, which is the prerequisite that lets the validator tell
a real metadata constraint from a search subject.
"""

from __future__ import annotations

from app.models.metadata_schema import (
    FieldType,
    MetadataField,
    MetadataSchema,
    operators_for_type,
)
from app.processors.schema_discovery import enrich_schema_with_values


def _bare_schema() -> MetadataSchema:
    def field(name: str, ftype: FieldType, multi: bool = False) -> MetadataField:
        return MetadataField(
            name=name,
            type=ftype,
            operators=operators_for_type(ftype, multi=multi),
            multi=multi,
        )

    return MetadataSchema.from_fields(
        [
            field("status", FieldType.STRING),
            field("techNotes", FieldType.STRING, multi=True),
            field("tags", FieldType.STRING, multi=True),
            field("date", FieldType.DATE),
        ]
    )


def test_low_cardinality_string_becomes_enumerable():
    values = {"status": ["Done", "In progress", "Done", "Not started", "Done"]}
    schema = enrich_schema_with_values(_bare_schema(), values)
    status = schema.get("status")
    assert status is not None
    assert status.is_enumerable
    assert set(status.allowed_values) == {"Done", "In progress", "Not started"}


def test_free_text_field_stays_non_enumerable():
    # Every value distinct -> free text, must not become a filterable enum.
    values = {"techNotes": [f"note number {i} about a unique topic" for i in range(30)]}
    schema = enrich_schema_with_values(_bare_schema(), values)
    tech = schema.get("techNotes")
    assert tech is not None
    assert not tech.is_enumerable
    assert tech.allowed_values == ()


def test_multi_select_low_cardinality_is_enumerable():
    values = {"tags": [["python", "rag"], ["python"], ["rag", "python"], ["rag"]]}
    schema = enrich_schema_with_values(_bare_schema(), values)
    tags = schema.get("tags")
    assert tags is not None
    assert tags.is_enumerable
    assert set(tags.allowed_values) == {"python", "rag"}


def test_date_field_never_enumerable():
    values = {"date": ["2026-07-13", "2026-07-14", "2026-07-13"]}
    schema = enrich_schema_with_values(_bare_schema(), values)
    d = schema.get("date")
    assert d is not None
    assert not d.is_enumerable


def test_opaque_id_relation_field_is_not_enumerable():
    # A Notion relation field holds page UUIDs, not categorical labels, even
    # when only a few distinct ids are sampled.
    uuids = [
        "382f37cd-5024-800a-b22e-eca13d16a73b",
        "386f37cd-5024-807f-9b60-e666cc9d3a4d",
        "390f37cd-5024-8044-8c49-ced6f93dca0d",
    ]
    values = {"techNotes": [[u] for u in uuids] * 3}
    schema = enrich_schema_with_values(_bare_schema(), values)
    tech = schema.get("techNotes")
    assert tech is not None
    assert not tech.is_enumerable
