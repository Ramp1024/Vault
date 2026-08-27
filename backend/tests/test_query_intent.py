"""Unit tests for the deterministic query-intent segmenter (Phase 3).

Lock in the three behaviors that close the LLM analyzer's remaining regressions:
lexical-lookup queries never produce a filter, real metadata queries produce the
canonical filter, and a value that merely appears without a field cue is not
turned into a constraint.
"""

from __future__ import annotations

from app.models.metadata_schema import (
    FieldType,
    MetadataField,
    MetadataSchema,
    operators_for_type,
)
from app.processors.query_intent import (
    ConstraintExtractor,
    DeterministicIntentAnalyzer,
)


def _schema() -> MetadataSchema:
    def string_field(name: str, values: tuple[str, ...]) -> MetadataField:
        return MetadataField(
            name=name,
            type=FieldType.STRING,
            operators=operators_for_type(FieldType.STRING),
            allowed_values=values,
        )

    return MetadataSchema.from_fields(
        [
            string_field(
                "category",
                ("AI Engineering", "System Design", "Front end System Design"),
            ),
            string_field(
                "leetcodeTopic",
                ("Arrays & Hashing", "Backtracking", "Graphs", "Heap/Priority Queue"),
            ),
            MetadataField(
                name="status",
                type=FieldType.STRING,
                operators=operators_for_type(FieldType.STRING),
                allowed_values=("Done", "In progress", "Not started"),
                value_aliases=(("Done", ("completed", "finished")),),
            ),
            MetadataField(
                name="techNotes",
                type=FieldType.STRING,
                operators=operators_for_type(FieldType.STRING, multi=True),
                multi=True,
            ),
        ]
    )


def _analyzer() -> DeterministicIntentAnalyzer:
    return DeterministicIntentAnalyzer(_schema(), default_top_k=10, min_confidence=1)


def _filters(query: str) -> set[str]:
    request = _analyzer().analyze(query)
    return {f"{f.field}={f.value}" for f in request.filters}


def test_lexical_lookup_never_filters():
    for query in [
        "Where did I mention BM25?",
        "What did I write about Prim's algorithm?",
        "What did I write about the backtracking pattern?",  # value present but lexical
        "When did I solve Pacific Atlantic Water Flow?",
        "Which day did I do Surrounded Regions?",
    ]:
        assert _filters(query) == set(), query


def test_metadata_query_produces_canonical_filter():
    assert _filters("What front-end system design notes do I have?") == {
        "category=Front end System Design"
    }
    assert _filters("What tasks have I not started yet?") == {"status=Not started"}
    assert _filters("What tasks are currently in progress?") == {"status=In progress"}
    assert _filters("Which heap or priority queue problems did I practice?") == {
        "leetcodeTopic=Heap/Priority Queue"
    }


def test_distinctive_value_matches_without_a_cue():
    # "Backtracking" is distinctive enough to stand alone.
    assert _filters("What backtracking leetcode problems have I solved?") == {
        "leetcodeTopic=Backtracking"
    }


def test_short_value_requires_a_field_cue():
    # "graph" is non-distinctive: only becomes a filter with a leetcode/problem cue.
    assert _filters("Which leetcode graph problems have I worked on?") == {
        "leetcodeTopic=Graphs"
    }
    assert _filters("What graph algorithms have I studied and taken notes on?") == set()


def test_subject_strips_the_matched_evidence():
    request = _analyzer().analyze("What tasks have I not started yet?")
    assert request.filters  # a status filter was produced
    subject = request.semantic_query.casefold()
    assert "not started" not in subject


def test_value_alias_resolves_paraphrase_with_a_cue():
    # "completed" is an LLM-generated synonym of Done; fires only with a field cue.
    assert _filters("What tasks have I completed?") == {"status=Done"}


def test_value_alias_does_not_fire_without_a_cue():
    # Same synonym, no status/task cue -> no filter (avoids over-firing).
    assert _filters("I completed the marathon this morning") == set()


def test_extractor_reports_evidence_and_confidence():
    matches = ConstraintExtractor(_schema()).extract(
        "Which leetcode graph problems have I worked on?"
    )
    graph = next(m for m in matches if m.field == "leetcodeTopic")
    assert graph.value == "Graphs"
    assert graph.evidence == "graph"
    assert graph.confidence >= 2  # value match + field cue
