from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.models.chunk import Chunk
from app.models.filter import Filter, Operator


def _normalize(value: Any) -> str:
    """Collapse a scalar to a comparable, case-insensitive string."""
    return str(value).strip().lower()


class MetadataFilterMatcher:
    """Apply storage-agnostic metadata filters to chunks in memory.

    Backends such as BM25 that cannot filter during retrieval use this to filter
    results afterwards. It mirrors ``QdrantFilterBuilder`` so filtering behaves
    consistently with vector retrieval: logical field names resolve to the
    chunk's nested ``properties`` map (the same layout the Qdrant payload uses),
    multiple filters are combined with AND (Qdrant ``must``), and a scalar filter
    against a list-valued field is treated as "contains".
    """

    PROPERTY_KEY = "properties"

    def matches(self, chunk: Chunk, filters: Sequence[Filter]) -> bool:
        """Return True if the chunk satisfies every filter."""
        return all(self._matches_one(chunk, f) for f in filters)

    def _matches_one(self, chunk: Chunk, f: Filter) -> bool:
        properties = chunk.metadata.get(self.PROPERTY_KEY)
        if not isinstance(properties, dict):
            return False

        stored = properties.get(f.field)
        if stored is None:
            return False

        if f.operator not in (Operator.EQUALS, Operator.CONTAINS):
            raise ValueError(f"Unsupported filter operator: {f.operator}")

        stored_values = self._as_value_set(stored)

        if isinstance(f.value, (list, tuple, set)):
            wanted = {_normalize(item) for item in f.value}
            return bool(wanted & stored_values)

        return _normalize(f.value) in stored_values

    @staticmethod
    def _as_value_set(stored: Any) -> set[str]:
        if isinstance(stored, (list, tuple, set)):
            return {_normalize(item) for item in stored}
        return {_normalize(stored)}
