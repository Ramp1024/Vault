from dataclasses import dataclass
from enum import Enum
from typing import Any


class Operator(str, Enum):
    """Comparison operators for metadata filters.

    The set is intentionally open for extension: new operators (e.g. GT, LT,
    GTE, LTE, IN) can be added here and handled by translators such as the
    QdrantFilterBuilder without changing the Filter or SearchRequest API.
    """

    EQUALS = "equals"
    CONTAINS = "contains"


@dataclass(frozen=True)
class Filter:
    """A single, storage-agnostic metadata filter condition.

    Attributes:
        field: Logical field name (independent of any payload layout).
        operator: How ``value`` should be matched against the field.
        value: The value to match.
    """

    field: str
    operator: Operator
    value: Any
