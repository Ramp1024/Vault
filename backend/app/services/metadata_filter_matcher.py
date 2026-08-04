from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from app.models.chunk import Chunk
from app.models.filter import Filter, Operator


def _normalize(value: Any) -> str:
    """Collapse a scalar to a comparable, case-insensitive string."""
    return str(value).strip().lower()


# Operators that compare ordered values (dates/numbers) rather than membership.
_RANGE_OPERATORS = frozenset(
    {Operator.GT, Operator.LT, Operator.GTE, Operator.LTE, Operator.BETWEEN}
)


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

        if f.operator in _RANGE_OPERATORS:
            return self._matches_range(stored, f)

        if f.operator not in (Operator.EQUALS, Operator.CONTAINS):
            raise ValueError(f"Unsupported filter operator: {f.operator}")

        stored_values = self._as_value_set(stored)

        if isinstance(f.value, (list, tuple, set)):
            wanted = {_normalize(item) for item in f.value}
            return bool(wanted & stored_values)

        return _normalize(f.value) in stored_values

    def _matches_range(self, stored: Any, f: Filter) -> bool:
        """Compare an ordered stored scalar against a range/between filter."""
        if isinstance(stored, (list, tuple, set)):
            # Ordered comparisons are undefined for multi-valued fields.
            return False

        if f.operator is Operator.BETWEEN:
            if not isinstance(f.value, (list, tuple)) or len(f.value) != 2:
                return False
            low = self._coerce(stored, f.value[0])
            high = self._coerce(stored, f.value[1])
            value = self._coerce(stored, stored)
            if low is None or high is None or value is None:
                return False
            return low <= value <= high

        bound = self._coerce(stored, f.value)
        value = self._coerce(stored, stored)
        if bound is None or value is None:
            return False
        if f.operator is Operator.GT:
            return value > bound
        if f.operator is Operator.GTE:
            return value >= bound
        if f.operator is Operator.LT:
            return value < bound
        return value <= bound  # LTE

    @staticmethod
    def _coerce(stored: Any, value: Any) -> Any:
        """Coerce ``value`` to a comparable form aligned with ``stored``.

        Dates compare as ISO strings, numbers as floats. Returns None when the
        value cannot be compared, so the filter simply fails closed.
        """
        if _looks_like_date(stored):
            return _as_date(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_value_set(stored: Any) -> set[str]:
        if isinstance(stored, (list, tuple, set)):
            return {_normalize(item) for item in stored}
        return {_normalize(stored)}


def _looks_like_date(value: Any) -> bool:
    return _as_date(value) is not None


def _as_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
