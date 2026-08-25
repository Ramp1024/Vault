from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.models.filter import Filter, Operator

# Coarse routing intents used by the query-analysis metrics. They describe what
# a correct analyzer should *do* with a query, independent of the eight surface
# categories: run pure vector search, apply a non-date metadata filter, or apply
# a temporal (date/authorship) filter.
VALID_INTENTS = ("semantic", "metadata", "temporal")
VALID_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class EvaluationCase:
    """A single golden retrieval expectation.

    Beyond the core retrieval expectation (``query`` -> ``expected_documents``),
    cases may carry optional annotations that drive the richer benchmark:
    ``difficulty`` for slicing, ``expected_filters`` as the ground-truth filter
    set a correct query analyzer should generate, and ``expected_intent`` as the
    coarse routing decision (semantic / metadata / temporal). The loader still
    ignores unknown keys so datasets remain forward-compatible.
    """

    id: str
    query: str
    expected_documents: tuple[str, ...]
    expected_chunks: tuple[str, ...] = ()
    category: str | None = None
    difficulty: str | None = None
    expected_filters: tuple[Filter, ...] = ()
    expected_intent: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluationCase":
        case_id = raw.get("id")
        query = raw.get("query")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each case needs a non-empty string 'id'")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case '{case_id}' needs a non-empty string 'query'")

        expected_documents = _string_tuple(
            raw.get("expected_documents", []), field_name="expected_documents"
        )
        expected_chunks = _string_tuple(
            raw.get("expected_chunks", []), field_name="expected_chunks"
        )
        category = raw.get("category")
        if category is not None and not isinstance(category, str):
            raise ValueError(f"Case '{case_id}' has a non-string 'category'")

        difficulty = raw.get("difficulty")
        if difficulty is not None and difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Case '{case_id}' has invalid 'difficulty' {difficulty!r}; "
                f"expected one of {VALID_DIFFICULTIES}"
            )

        expected_filters = _parse_filters(
            raw.get("expected_filters", []), case_id=case_id
        )

        intent = raw.get("expected_intent")
        if intent is not None and intent not in VALID_INTENTS:
            raise ValueError(
                f"Case '{case_id}' has invalid 'expected_intent' {intent!r}; "
                f"expected one of {VALID_INTENTS}"
            )
        if intent is None:
            intent = _derive_intent(expected_filters)

        # Unknown keys (future answer/citation fields, human annotations) are
        # intentionally ignored so datasets can be extended without code changes.
        return cls(
            id=case_id,
            query=query,
            expected_documents=expected_documents,
            expected_chunks=expected_chunks,
            category=category,
            difficulty=difficulty,
            expected_filters=expected_filters,
            expected_intent=intent,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "query": self.query}
        if self.category is not None:
            data["category"] = self.category
        if self.difficulty is not None:
            data["difficulty"] = self.difficulty
        data["expected_documents"] = list(self.expected_documents)
        data["expected_chunks"] = list(self.expected_chunks)
        if self.expected_filters:
            data["expected_filters"] = [
                {
                    "field": f.field,
                    "operator": f.operator.value,
                    "value": f.value,
                }
                for f in self.expected_filters
            ]
        if self.expected_intent is not None:
            data["expected_intent"] = self.expected_intent
        return data


@dataclass(frozen=True)
class EvaluationDataset:
    """An ordered collection of golden evaluation cases."""

    cases: tuple[EvaluationCase, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[EvaluationCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    @classmethod
    def from_list(cls, raw: list[dict[str, Any]]) -> "EvaluationDataset":
        if not isinstance(raw, list):
            raise ValueError("Dataset must be a JSON array of cases")
        cases = tuple(EvaluationCase.from_dict(item) for item in raw)
        ids = [c.id for c in cases]
        if len(set(ids)) != len(ids):
            raise ValueError("Dataset contains duplicate case ids")
        return cls(cases=cases)

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_list(raw)

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [case.to_dict() for case in self.cases]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"'{field_name}' must be a list of non-empty strings")
    return tuple(value)


# Fields that carry temporal meaning; a filter on any of them marks a case (or an
# analyzer's output) as a "temporal" routing intent rather than plain metadata.
DATE_FIELDS = frozenset({"date", "last_edited_time", "created_time"})


def _parse_filters(value: Any, *, case_id: str) -> tuple[Filter, ...]:
    """Parse the ``expected_filters`` array into domain ``Filter`` objects."""
    if not isinstance(value, list):
        raise ValueError(f"Case '{case_id}' has non-list 'expected_filters'")
    filters: list[Filter] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Case '{case_id}' filter entries must be objects"
            )
        field_name = entry.get("field")
        operator = entry.get("operator")
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(f"Case '{case_id}' filter needs a string 'field'")
        try:
            op = Operator(operator)
        except ValueError as exc:
            raise ValueError(
                f"Case '{case_id}' filter has invalid operator {operator!r}"
            ) from exc
        filters.append(
            Filter(field=field_name, operator=op, value=entry.get("value"))
        )
    return tuple(filters)


def _derive_intent(filters: tuple[Filter, ...]) -> str:
    """Infer the coarse routing intent implied by a case's expected filters."""
    if any(f.field in DATE_FIELDS for f in filters):
        return "temporal"
    if filters:
        return "metadata"
    return "semantic"
