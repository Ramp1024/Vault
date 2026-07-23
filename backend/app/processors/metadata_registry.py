from __future__ import annotations

import re
from collections.abc import Iterable

from app.core.text import to_camel_case


def _normalize(token: str) -> str:
    """Collapse a token to a comparable key: lowercased alphanumerics only."""
    return "".join(ch for ch in token.lower() if ch.isalnum())


def _decamel(name: str) -> str:
    """Turn a camelCase field name into space-separated words (``leetcodeTopic`` -> ``leetcode topic``)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return spaced.replace("_", " ").strip().lower()



class MetadataRegistry:
    """A configurable registry of recognized metadata fields.

    The registry maps user-facing surface forms (e.g. ``"leetcode topic"``) to a
    canonical logical field name (e.g. ``"leetcodeTopic"``). It is intentionally
    connector-agnostic: it can be seeded from a static list or derived from the
    property names actually present in the index, so new fields require no code
    changes to the analyzer.
    """

    def __init__(self) -> None:
        self._canonical_by_norm: dict[str, str] = {}
        self._surfaces: set[str] = set()
        self._multi: set[str] = set()
        self._kinds: dict[str, str] = {}

    def register(
        self,
        raw_name: str,
        *,
        multi: bool = False,
        kind: str = "text",
        extra_aliases: Iterable[str] = (),
    ) -> None:
        """Register a field by its raw (human) name.

        Args:
            raw_name: The source property name, e.g. ``"Leetcode Topic"``.
            multi: Whether the field holds multiple values (list-like), which
                selects a ``CONTAINS`` rather than ``EQUALS`` operator.
            kind: The value kind (``"text"`` or ``"date"``), which selects how
                the analyzer extracts and normalizes the field's value.
            extra_aliases: Additional surface forms that should resolve here.
        """
        canonical = to_camel_case(raw_name)
        if not canonical:
            return

        surfaces = {raw_name.strip().lower(), canonical.lower()}
        surfaces.update(alias.lower() for alias in extra_aliases)

        for surface in surfaces:
            if not surface:
                continue
            self._surfaces.add(surface)
            self._canonical_by_norm[_normalize(surface)] = canonical

        self._canonical_by_norm[_normalize(canonical)] = canonical
        self._kinds[canonical] = kind
        if multi:
            self._multi.add(canonical)

    def resolve(self, token: str) -> str | None:
        """Resolve a matched surface form to its canonical field name."""
        return self._canonical_by_norm.get(_normalize(token))

    def is_multi(self, canonical: str) -> bool:
        """Return True if the canonical field holds multiple values."""
        return canonical in self._multi

    def kind_of(self, canonical: str) -> str:
        """Return the value kind for the canonical field (default ``"text"``)."""
        return self._kinds.get(canonical, "text")

    def surfaces(self) -> list[str]:
        """Return known surface forms, longest first (for greedy matching)."""
        return sorted(self._surfaces, key=len, reverse=True)

    @classmethod
    def from_property_names(
        cls, names: Iterable[str], multi_fields: Iterable[str] = ()
    ) -> "MetadataRegistry":
        """Build a registry from raw property names (e.g. discovered from the index)."""
        multi_canonical = {to_camel_case(name) for name in multi_fields}
        registry = cls()
        for name in names:
            canonical = to_camel_case(name)
            kind = "date" if "date" in name.lower() else "text"
            registry.register(name, multi=canonical in multi_canonical, kind=kind)
        return registry

    @classmethod
    def from_indexed_fields(
        cls, fields: Iterable[str], multi_fields: Iterable[str] = ()
    ) -> "MetadataRegistry":
        """Build a registry from canonical (camelCase) indexed payload keys.

        The indexed keys are already canonical, so they are de-camelCased into
        human surface forms (``leetcodeTopic`` -> ``leetcode topic``) before
        registration; ``to_camel_case`` then reconstructs the identical canonical
        name, keeping the registry's field names aligned with the payload keys
        used for filtering.
        """
        return cls.from_property_names(
            (_decamel(field) for field in fields),
            multi_fields=[_decamel(field) for field in multi_fields],
        )



def default_metadata_registry() -> MetadataRegistry:
    """Registry seeded with the fields currently emitted by the Notion connector.

    Additional generic fields common across connectors are included so the
    analyzer stays useful beyond a single source.
    """
    registry = MetadataRegistry()
    for name in ("Leetcode", "Leetcode Topic", "Personal Win", "Work Win", "Category"):
        registry.register(name)
    registry.register("Date", kind="date")
    registry.register("Status")
    registry.register("Project")
    registry.register("Team")
    registry.register("Tags", multi=True, extra_aliases=("tag",))
    return registry
