"""Unit tests for the schema-driven, timezone-aware temporal pipeline.

Cover the four moving parts:
  * timezone helpers (local day <-> UTC instants),
  * the tz-aware BM25 filter matcher (the boundary-note bug),
  * the Qdrant range builder's UTC conversion for activity dates,
  * deterministic temporal detection and the fail-safe field selector.
"""

from __future__ import annotations

from datetime import date

from app.core.clock import local_day_bounds_to_utc, to_local_day
from app.models.chunk import Chunk
from app.models.filter import Filter, Operator
from app.models.metadata_schema import (
    TEMPORAL_ACTIVITY,
    TEMPORAL_CONTENT,
    FieldType,
    MetadataField,
    MetadataSchema,
    operators_for_type,
)
from app.processors.query_intent import DeterministicIntentAnalyzer
from app.processors.temporal_field_selector import TemporalFieldSelector
from app.processors.temporal_query import detect_temporal, detect_temporal_range
from app.services.metadata_filter_matcher import MetadataFilterMatcher
from app.services.qdrant_filter_builder import QdrantFilterBuilder

TODAY = date(2026, 8, 25)


# ---------------------------------------------------------------------------
# timezone helpers
# ---------------------------------------------------------------------------
def test_local_day_bounds_to_utc_offsets_by_timezone():
    low, high = local_day_bounds_to_utc(date(2026, 7, 13), date(2026, 7, 19))
    # Asia/Kolkata is +5:30, so the local day starts the previous UTC evening.
    assert low == "2026-07-12T18:30:00+00:00"
    assert high.startswith("2026-07-19T18:29:59")


def test_to_local_day_shifts_utc_evening_into_next_day():
    assert to_local_day("2026-07-12T18:30:00.000Z") == date(2026, 7, 13)
    assert to_local_day("2026-07-12T12:00:00.000Z") == date(2026, 7, 12)
    assert to_local_day("2026-07-13") == date(2026, 7, 13)


# ---------------------------------------------------------------------------
# BM25 matcher — the boundary-note bug
# ---------------------------------------------------------------------------
def _chunk(last_edited: str) -> Chunk:
    return Chunk(
        id="c1",
        document_id="d1",
        document_title="t",
        content="body",
        chunk_index=0,
        metadata={"last_edited_time": last_edited},
    )


def _week_filter() -> Filter:
    return Filter(
        field="last_edited_time",
        operator=Operator.BETWEEN,
        value=["2026-07-13", "2026-07-19"],
    )


def test_matcher_includes_utc_evening_note_in_local_week():
    # 2026-07-12 18:30Z == 2026-07-13 00:00 IST -> inside the July 13 week.
    matcher = MetadataFilterMatcher()
    assert matcher.matches(_chunk("2026-07-12T18:30:00.000Z"), [_week_filter()])


def test_matcher_excludes_note_before_local_week():
    # 2026-07-12 12:00Z == 2026-07-12 17:30 IST -> still July 12, excluded.
    matcher = MetadataFilterMatcher()
    assert not matcher.matches(_chunk("2026-07-12T12:00:00.000Z"), [_week_filter()])


# ---------------------------------------------------------------------------
# Qdrant builder — UTC conversion for activity dates
# ---------------------------------------------------------------------------
def test_builder_converts_activity_date_bounds_to_utc_instants():
    qfilter = QdrantFilterBuilder().build([_week_filter()])
    rng = qfilter.must[0].range
    # Qdrant coerces the RFC3339 strings into datetimes; the local IST day start
    # (2026-07-13 00:00 IST) is the UTC instant 2026-07-12 18:30Z.
    assert (rng.gte.year, rng.gte.month, rng.gte.day) == (2026, 7, 12)
    assert (rng.gte.hour, rng.gte.minute) == (18, 30)
    assert rng.gte.utcoffset().total_seconds() == 0
    assert (rng.lte.month, rng.lte.day, rng.lte.hour) == (7, 19, 18)


def test_builder_keeps_content_date_bounds_as_calendar_days():
    content = Filter(
        field="date", operator=Operator.BETWEEN, value=["2026-07-13", "2026-07-19"]
    )
    rng = QdrantFilterBuilder().build([content]).must[0].range
    # Content dates stay calendar days (midnight, no timezone shift).
    assert (rng.gte.year, rng.gte.month, rng.gte.day) == (2026, 7, 13)
    assert (rng.gte.hour, rng.gte.minute) == (0, 0)
    assert (rng.lte.year, rng.lte.month, rng.lte.day) == (2026, 7, 19)


# ---------------------------------------------------------------------------
# Deterministic temporal detection
# ---------------------------------------------------------------------------
def test_detect_week_of_and_month():
    assert detect_temporal_range("work on during the week of July 13?", TODAY) == (
        "2026-07-13",
        "2026-07-19",
    )
    assert detect_temporal_range("what I practiced in August?", TODAY) == (
        "2026-08-01",
        "2026-08-31",
    )


def test_detect_returns_none_without_temporal_expression():
    assert detect_temporal_range("Where did I mention BM25?", TODAY) is None


# ---------------------------------------------------------------------------
# Explicit calendar dates (single-day resolution)
# ---------------------------------------------------------------------------
def test_detect_explicit_calendar_date_is_single_day():
    for query in [
        "August 4",
        "What have I written about August 4",
        "August 4th",
        "4 August",
        "on 4th of August",
        "August 4, 2026",
    ]:
        assert detect_temporal_range(query, TODAY) == ("2026-08-04", "2026-08-04"), query


def test_detect_iso_date_is_single_day():
    assert detect_temporal_range("notes from 2026-08-04", TODAY) == (
        "2026-08-04",
        "2026-08-04",
    )


def test_detect_abbreviated_month_is_single_day():
    # Abbreviated month names (with or without a trailing period) resolve too.
    for query in [
        "summarise what i did on Aug 4",
        "Aug 4",
        "Aug. 4",
        "4 Aug",
    ]:
        assert detect_temporal_range(query, TODAY) == ("2026-08-04", "2026-08-04"), query


def test_detect_reports_day_unit_for_explicit_date():
    match = detect_temporal("What did I do on August 4?", TODAY)
    assert match is not None
    assert match.unit == "day"
    assert (match.low, match.high) == ("2026-08-04", "2026-08-04")


def test_bare_month_stays_a_month_range_not_a_day():
    # "August 2026" (the whole month) must not be mis-read as "August 20".
    assert detect_temporal_range("what I practiced in August 2026", TODAY) == (
        "2026-08-01",
        "2026-08-31",
    )
    assert detect_temporal("in August", TODAY).unit == "month"


# ---------------------------------------------------------------------------
# Temporal field selector — fail-safe fallback
# ---------------------------------------------------------------------------
def _date_field(name: str, role: str) -> MetadataField:
    return MetadataField(
        name=name,
        type=FieldType.DATE,
        operators=operators_for_type(FieldType.DATE),
        temporal_role=role,
    )


def test_selector_single_candidate_is_deterministic():
    schema = MetadataSchema.from_fields([_date_field("last_edited_time", TEMPORAL_ACTIVITY)])
    selection = TemporalFieldSelector(schema).select("week of July 13")
    assert selection.field == "last_edited_time"
    assert selection.confidence == 1.0


def test_selector_prefers_activity_when_ambiguous_and_no_llm():
    schema = MetadataSchema.from_fields(
        [_date_field("date", TEMPORAL_CONTENT), _date_field("last_edited_time", TEMPORAL_ACTIVITY)]
    )
    selection = TemporalFieldSelector(schema).select("what did I work on last week")
    assert selection.field == "last_edited_time"


def test_selector_drops_filter_when_no_date_fields():
    schema = MetadataSchema.from_fields(
        [
            MetadataField(
                name="status",
                type=FieldType.STRING,
                operators=operators_for_type(FieldType.STRING),
                allowed_values=("Done",),
            )
        ]
    )
    selection = TemporalFieldSelector(schema).select("last week")
    assert not selection.selected
    assert selection.field is None


def test_selector_prefers_requested_role_when_present():
    schema = MetadataSchema.from_fields(
        [
            _date_field("date", TEMPORAL_CONTENT),
            _date_field("last_edited_time", TEMPORAL_ACTIVITY),
        ]
    )
    selection = TemporalFieldSelector(schema).select(
        "August 4", prefer_role=TEMPORAL_CONTENT
    )
    assert selection.field == "date"


def test_selector_prefer_role_falls_back_when_role_absent():
    # No content-date field exists; the request degrades to the lone candidate.
    schema = MetadataSchema.from_fields(
        [_date_field("last_edited_time", TEMPORAL_ACTIVITY)]
    )
    selection = TemporalFieldSelector(schema).select(
        "August 4", prefer_role=TEMPORAL_CONTENT
    )
    assert selection.field == "last_edited_time"


# ---------------------------------------------------------------------------
# Deterministic analyzer wires temporal onto the selected field
# ---------------------------------------------------------------------------
def test_deterministic_analyzer_adds_temporal_filter_on_activity_field():
    schema = MetadataSchema.from_fields(
        [
            _date_field("date", TEMPORAL_CONTENT),
            _date_field("last_edited_time", TEMPORAL_ACTIVITY),
        ]
    )
    request = DeterministicIntentAnalyzer(schema, default_top_k=10).analyze(
        "What did I work on during the week of July 13?"
    )
    date_filters = [f for f in request.filters if f.field == "last_edited_time"]
    assert len(date_filters) == 1
    assert date_filters[0].operator is Operator.BETWEEN


def test_deterministic_analyzer_routes_explicit_date_to_content_field():
    # An explicit calendar date means the note's content date, so it filters the
    # content-date field (not the activity date) and resolves to a single day.
    schema = MetadataSchema.from_fields(
        [
            _date_field("date", TEMPORAL_CONTENT),
            _date_field("last_edited_time", TEMPORAL_ACTIVITY),
        ]
    )
    request = DeterministicIntentAnalyzer(schema, default_top_k=10).analyze(
        "What have I written about August 4?"
    )
    assert not any(f.field == "last_edited_time" for f in request.filters)
    content_filters = [f for f in request.filters if f.field == "date"]
    assert len(content_filters) == 1
    assert content_filters[0].operator is Operator.BETWEEN
    assert content_filters[0].value == ["2026-08-04", "2026-08-04"]
