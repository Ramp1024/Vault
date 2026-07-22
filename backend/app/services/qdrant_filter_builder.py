from __future__ import annotations

from collections.abc import Sequence

from qdrant_client.http.models import (
    FieldCondition,
    Filter as QdrantFilter,
    MatchAny,
    MatchValue,
)

from app.models.filter import Filter, Operator


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

        if isinstance(f.value, (list, tuple, set)):
            return FieldCondition(key=key, match=MatchAny(any=list(f.value)))

        if f.operator in (Operator.EQUALS, Operator.CONTAINS):
            # For array payload fields Qdrant treats a scalar MatchValue as
            # "array contains value", which covers CONTAINS; for scalar fields it
            # is exact equality.
            return FieldCondition(key=key, match=MatchValue(value=f.value))

        raise ValueError(f"Unsupported filter operator: {f.operator}")
