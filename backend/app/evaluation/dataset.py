from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class EvaluationCase:
    """A single golden retrieval expectation.

    Only retrieval fields are defined today. The schema is deliberately open for
    extension: the loader ignores unknown keys, so future answer/citation fields
    (e.g. ``expected_answer``, ``required_citations``, ``evaluation_notes``) can
    be added to the dataset files and this dataclass later without breaking
    existing datasets or the runner.
    """

    id: str
    query: str
    expected_documents: tuple[str, ...]
    expected_chunks: tuple[str, ...] = ()
    category: str | None = None

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

        # Unknown keys (future answer/citation fields, human annotations) are
        # intentionally ignored so datasets can be extended without code changes.
        return cls(
            id=case_id,
            query=query,
            expected_documents=expected_documents,
            expected_chunks=expected_chunks,
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "query": self.query}
        if self.category is not None:
            data["category"] = self.category
        data["expected_documents"] = list(self.expected_documents)
        data["expected_chunks"] = list(self.expected_chunks)
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
