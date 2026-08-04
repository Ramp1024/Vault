from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from qdrant_client.http.models import (
    DatetimeRange,
    FieldCondition,
    Filter as QdrantFilter,
    MatchAny,
    MatchValue,
    Range,
)

from app.models.filter import Filter, Operator

# Operators that select a bounded/ordered slice of an ordered field.
_RANGE_OPERATORS = frozenset(
    {Operator.GT, Operator.LT, Operator.GTE, Operator.LTE, Operator.BETWEEN}
)


class QdrantFilterBuilder:
    """Translate storage-agnostic filters into Qdrant payload filters.

    This is the single place that knows how logical field names map onto the
    Qdrant payload layout. Page properties are stored under a nested
    ``properties`` object, so a logical field ``leetcodeTopic`` maps to the
    payload path ``properties.leetcodeTopic``.
    """

    PROPERTY_PREFIX = "properties"

    def build(self, filters: Sequence[Filter]) -> QdrantFilter | None:
        """Build a Qdrant filter from generic filters, or None if there are none."""
        if not filters:
            return None

        conditions = [self._condition(f) for f in filters]
        return QdrantFilter(must=conditions)

    def _payload_key(self, field: str) -> str:
        return f"{self.PROPERTY_PREFIX}.{field}"

    def _condition(self, f: Filter) -> FieldCondition:
        key = self._payload_key(f.field)

        if f.operator in _RANGE_OPERATORS:
            return FieldCondition(key=key, range=self._range(f))

        if isinstance(f.value, (list, tuple, set)):
            return FieldCondition(key=key, match=MatchAny(any=list(f.value)))

        if f.operator in (Operator.EQUALS, Operator.CONTAINS):
            # For array payload fields Qdrant treats a scalar MatchValue as
            # "array contains value", which covers CONTAINS; for scalar fields it
            # is exact equality.
            return FieldCondition(key=key, match=MatchValue(value=f.value))

        raise ValueError(f"Unsupported filter operator: {f.operator}")

    def _range(self, f: Filter) -> Range | DatetimeRange:
        """Translate a range/between filter into a Qdrant Range or DatetimeRange.

        Date-valued fields use ``DatetimeRange`` (comparing ISO date strings);
        numeric fields use the numeric ``Range``. ``BETWEEN`` expects a
        two-element ``[low, high]`` value and maps to an inclusive gte/lte range.
        """
        if f.operator is Operator.BETWEEN:
            low, high = self._between_bounds(f.value)
            return self._bounded_range(low, high, inclusive=True)

        bound = f.value
        if f.operator is Operator.GT:
            return self._bounded_range(bound, None, inclusive=False)
        if f.operator is Operator.GTE:
            return self._bounded_range(bound, None, inclusive=True)
        if f.operator is Operator.LT:
            return self._bounded_range(None, bound, inclusive=False)
        # LTE
        return self._bounded_range(None, bound, inclusive=True)

    def _bounded_range(
        self, low: object, high: object, *, inclusive: bool
    ) -> Range | DatetimeRange:
        # Choose the datetime range when either bound is a date string; Qdrant's
        # DatetimeRange compares RFC3339/ISO values, which the payload stores.
        if self._is_date(low) or self._is_date(high):
            if inclusive:
                return DatetimeRange(gte=low, lte=high)
            return DatetimeRange(gt=low, lt=high)
        low_num = self._as_number(low)
        high_num = self._as_number(high)
        if inclusive:
            return Range(gte=low_num, lte=high_num)
        return Range(gt=low_num, lt=high_num)

    @staticmethod
    def _between_bounds(value: object) -> tuple[object, object]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return value[0], value[1]
        raise ValueError("BETWEEN filter requires a two-element [low, high] value")

    @staticmethod
    def _is_date(value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            date.fromisoformat(value.strip()[:10])
            return True
        except ValueError:
            return False

    @staticmethod
    def _as_number(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Range bounds cannot be boolean")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except ValueError as exc:
            raise ValueError(f"Range bound is not numeric: {value!r}") from exc
