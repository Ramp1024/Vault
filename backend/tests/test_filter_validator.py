"""Unit tests for value-level filter validation (Phase 2 of the eligibility work).

These lock in the two defenses that eliminate the LLM analyzer's
"unnecessary filtering" regressions:

* Enumerable fields accept only known values (snapped to canonical casing);
  unknown values are dropped.
* Free-text (non-enumerable) string fields are never filterable — their content
  is a search subject, not a metadata constraint.
"""

from __future__ import annotations

from app.models.metadata_schema import (
    FieldType,
    MetadataField,
    MetadataSchema,
    operators_for_type,
)
from app.processors.filter_validator import FilterValidator


def _schema() -> MetadataSchema:
    return MetadataSchema.from_fields(
        [
            MetadataField(
                name="status",
                type=FieldType.STRING,
                operators=operators_for_type(FieldType.STRING),
                allowed_values=("Done", "In progress", "Not started"),
            ),
            MetadataField(
                name="leetcodeTopic",
                type=FieldType.STRING,
                operators=operators_for_type(FieldType.STRING),
                allowed_values=("Graphs", "Backtracking", "Heap/Priority Queue"),
            ),
            # Free-text multi field: no allowed values -> not enumerable.
            MetadataField(
                name="techNotes",
                type=FieldType.STRING,
                operators=operators_for_type(FieldType.STRING, multi=True),
                multi=True,
            ),
            MetadataField(
                name="date",
                type=FieldType.DATE,
                operators=operators_for_type(FieldType.DATE),
            ),
            MetadataField(
                name="leetcode47",
                type=FieldType.NUMBER,
                operators=operators_for_type(FieldType.NUMBER),
            ),
        ]
    )


def _validate(raw: list[dict]):
    return FilterValidator(_schema()).validate(raw)


def test_known_enum_value_is_accepted():
    filters = _validate([{"field": "leetcodeTopic", "operator": "=", "value": "Graphs"}])
    assert len(filters) == 1
    assert filters[0].field == "leetcodeTopic"
    assert filters[0].value == "Graphs"


def test_enum_value_is_snapped_to_canonical_casing():
    filters = _validate([{"field": "status", "operator": "=", "value": "done"}])
    assert len(filters) == 1
    assert filters[0].value == "Done"


def test_unknown_enum_value_is_dropped():
    # "writing" / "discussed" are the classic LLM hallucinations.
    filters = _validate(
        [
            {"field": "status", "operator": "=", "value": "discussed"},
            {"field": "leetcodeTopic", "operator": "=", "value": "writing"},
        ]
    )
    assert filters == []


def test_free_text_field_filter_is_rejected():
    # The BM25 regression: subject-as-filter on a free-text field.
    filters = _validate(
        [{"field": "techNotes", "operator": "contains", "value": "BM25"}]
    )
    assert filters == []


def test_underscored_value_does_not_match_spaced_enum():
    # "in_progress" must not silently match "In progress".
    filters = _validate([{"field": "status", "operator": "=", "value": "in_progress"}])
    assert filters == []


def test_number_and_date_fields_are_unaffected_by_membership():
    filters = _validate(
        [
            {"field": "leetcode47", "operator": "=", "value": 3},
            {"field": "date", "operator": "=", "value": "2026-07-13"},
        ]
    )
    fields = {f.field for f in filters}
    assert fields == {"leetcode47", "date"}
