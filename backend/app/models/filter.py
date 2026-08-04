from dataclasses import dataclass
from enum import Enum
from typing import Any


class Operator(str, Enum):
    """Comparison operators for metadata filters.

    The set is intentionally open for extension: new operators can be added here
    and handled by translators such as the QdrantFilterBuilder without changing
    the Filter or SearchRequest API. The range operators (GT/LT/GTE/LTE/BETWEEN)
    support ordered fields such as dates and numbers, while EQUALS/CONTAINS cover
    scalar and membership matching.
    """

    EQUALS = "equals"
    CONTAINS = "contains"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"


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
