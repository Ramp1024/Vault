from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.core.text import to_camel_case
from app.models.property_role import PropertyRole

# Heuristics for the ambiguous-type fallback. A value only reaches this path
# when the connector-native type is unknown/unmapped, so the thresholds only
# need to separate "short categorical" from "long-form prose".
_PROSE_CHAR_THRESHOLD = 80
_PROSE_WORD_THRESHOLD = 12


class PropertyRoleClassifier:
    """Classify a source property's :class:`PropertyRole` during ingestion.

    Role and datatype are separate concerns: this classifier answers *how should
    this property participate in retrieval?* independently of *how should it be
    queried?*. Classification follows a strict precedence:

    1. **User overrides** (highest): an explicit per-property role always wins.
    2. **Connector-native type**: a configurable map from the source's native
       property type (e.g. Notion ``date``/``rich_text``) to a role.
    3. **Property characteristics**: only for ambiguous/unmapped types — long
       free text defaults to ``SEMANTIC``, short categorical values to
       ``FILTERABLE``.

    The classifier itself is connector-agnostic; connectors supply their own
    ``type_role_map`` so no source-specific knowledge is hardcoded here.
    """

    def __init__(
        self,
        type_role_map: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        default_role: PropertyRole = PropertyRole.FILTERABLE,
    ) -> None:
        self._type_role_map = self._normalize_type_map(type_role_map)
        self._overrides = self._normalize_overrides(overrides)
        self._default_role = default_role

    def classify(
        self,
        *,
        name: str,
        key: str | None = None,
        native_type: str | None = None,
        value: Any = None,
    ) -> PropertyRole:
        """Return the :class:`PropertyRole` for a single property.

        Args:
            name: The property's original (human-readable) name.
            key: The canonical camelCase key; derived from ``name`` if omitted.
            native_type: The connector-native property type, if known.
            value: A sample value, used only for the characteristics fallback.
        """
        canonical = key or to_camel_case(name)

        override = self._lookup_override(name, canonical)
        if override is not None:
            return override

        if native_type is not None:
            role = self._type_role_map.get(str(native_type).strip().lower())
            if role is not None:
                return role

        return self._classify_by_characteristics(value)

    def _classify_by_characteristics(self, value: Any) -> PropertyRole:
        """Fallback classification for ambiguous or unmapped property types."""
        if value is None:
            return self._default_role
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return PropertyRole.FILTERABLE
        if isinstance(value, (list, tuple, set)):
            # Multi-valued fields are treated as short categorical tags.
            return PropertyRole.FILTERABLE
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return self._default_role
            if len(text) >= _PROSE_CHAR_THRESHOLD or len(text.split()) >= _PROSE_WORD_THRESHOLD:
                return PropertyRole.SEMANTIC
            return PropertyRole.FILTERABLE
        return self._default_role

    def _lookup_override(self, name: str, canonical: str) -> PropertyRole | None:
        candidates = [canonical]
        if name:
            candidates.append(to_camel_case(name))
            candidates.append(name.strip().lower())
        for candidate in candidates:
            if candidate and candidate in self._overrides:
                return self._overrides[candidate]
        return None

    @staticmethod
    def _normalize_type_map(
        type_role_map: Mapping[str, Any] | None,
    ) -> dict[str, PropertyRole]:
        normalized: dict[str, PropertyRole] = {}
        for raw_type, raw_role in (type_role_map or {}).items():
            role = PropertyRole.from_value(raw_role)
            if role is None:
                continue
            normalized[str(raw_type).strip().lower()] = role
        return normalized

    @staticmethod
    def _normalize_overrides(
        overrides: Mapping[str, Any] | None,
    ) -> dict[str, PropertyRole]:
        normalized: dict[str, PropertyRole] = {}
        for raw_name, raw_role in (overrides or {}).items():
            role = PropertyRole.from_value(raw_role)
            if role is None:
                continue
            # Index each override under both its camelCase key and its raw
            # lowercased name so users can write either form in configuration.
            normalized[to_camel_case(raw_name)] = role
            normalized[str(raw_name).strip().lower()] = role
        return normalized
