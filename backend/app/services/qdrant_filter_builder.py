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

from app.core.clock import local_day_bounds_to_utc
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

    # System timestamps are stored at the payload top level (spread from chunk
    # metadata), not under the nested ``properties`` object, so they must not be
    # prefixed like page properties.
    SYSTEM_FIELDS = frozenset({"last_edited_time", "created_time"})

    def build(self, filters: Sequence[Filter]) -> QdrantFilter | None:
        """Build a Qdrant filter from generic filters, or None if there are none."""
        if not filters:
            return None

        conditions = [self._condition(f) for f in filters]
        return QdrantFilter(must=conditions)

    def _payload_key(self, field: str) -> str:
        if field in self.SYSTEM_FIELDS:
            return field
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
        For system activity-date fields (stored as UTC timestamps) a local-day
        bound is converted to the matching UTC instant, so a note edited near
        midnight is matched against the day the user means, not the UTC day.
        """
        activity = f.field in self.SYSTEM_FIELDS
        if f.operator is Operator.BETWEEN:
            low, high = self._between_bounds(f.value)
            return self._bounded_range(low, high, inclusive=True, activity=activity)

        bound = f.value
        if f.operator is Operator.GT:
            return self._bounded_range(bound, None, inclusive=False, activity=activity)
        if f.operator is Operator.GTE:
            return self._bounded_range(bound, None, inclusive=True, activity=activity)
        if f.operator is Operator.LT:
            return self._bounded_range(None, bound, inclusive=False, activity=activity)
        # LTE
        return self._bounded_range(None, bound, inclusive=True, activity=activity)

    def _bounded_range(
        self, low: object, high: object, *, inclusive: bool, activity: bool = False
    ) -> Range | DatetimeRange:
        # Choose the datetime range when either bound is a date string; Qdrant's
        # DatetimeRange compares RFC3339/ISO values, which the payload stores.
        if self._is_date(low) or self._is_date(high):
            low_v, high_v = self._date_bounds(low, high, activity)
            if inclusive:
                return DatetimeRange(gte=low_v, lte=high_v)
            return DatetimeRange(gt=low_v, lt=high_v)
        low_num = self._as_number(low)
        high_num = self._as_number(high)
        if inclusive:
            return Range(gte=low_num, lte=high_num)
        return Range(gt=low_num, lt=high_num)

    @staticmethod
    def _date_bounds(
        low: object, high: object, activity: bool
    ) -> tuple[object, object]:
        """Return comparison bounds for a date range.

        Content dates are stored as calendar days, so their date-only bounds are
        used as-is. Activity dates are stored as UTC timestamps, so each local-day
        bound is expanded to the UTC instant at that day's start (low) or end
        (high).
        """
        if not activity:
            return low, high
        low_v = high_v = None
        if isinstance(low, str):
            day = date.fromisoformat(low.strip()[:10])
            low_v = local_day_bounds_to_utc(day, day)[0]
        if isinstance(high, str):
            day = date.fromisoformat(high.strip()[:10])
            high_v = local_day_bounds_to_utc(day, day)[1]
        return low_v, high_v

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
