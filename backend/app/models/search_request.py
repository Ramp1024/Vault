from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.models.filter import Filter


@dataclass(frozen=True)
class SearchRequest:
    """A structured retrieval request produced from a raw user query.

    Attributes:
        semantic_query: The free-text portion used for vector similarity search.
        filters: Structured, storage-agnostic metadata filter conditions.
        top_k: Maximum number of results to retrieve.
    """

    semantic_query: str
    filters: List[Filter] = field(default_factory=list)
    top_k: int = 5
