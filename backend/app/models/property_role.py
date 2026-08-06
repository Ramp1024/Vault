from __future__ import annotations

from enum import Enum


class PropertyRole(Enum):
    """How a source property participates in retrieval.

    A property's *role* is independent of its datatype. It determines whether a
    property becomes a filterable metadata attribute, semantic search content,
    or is discarded entirely:

    - ``FILTERABLE``: contributes to the :class:`MetadataSchema`, undergoes
      datatype inference, generates filter definitions and Qdrant payload
      indexes, and is exposed to the LLM intent analyzer.
    - ``SEMANTIC``: does not appear in the schema and never becomes a filter;
      its content is merged into the page body before chunking so it is part of
      the semantic search corpus.
    - ``IGNORE``: participates in neither filtering nor embedding.
    """

    FILTERABLE = "filterable"
    SEMANTIC = "semantic"
    IGNORE = "ignore"

    @classmethod
    def from_value(cls, value: object) -> "PropertyRole | None":
        """Coerce a raw value into a :class:`PropertyRole`, or ``None``.

        Accepts an existing :class:`PropertyRole` or a case-insensitive string
        (``"filterable"``, ``"semantic"``, ``"ignore"``). Unknown values return
        ``None`` so callers can fall back to their own defaults.
        """
        if isinstance(value, PropertyRole):
            return value
        if not isinstance(value, str):
            return None
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None
