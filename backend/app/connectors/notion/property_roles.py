from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.property_role import PropertyRole
from app.processors.property_role_classifier import PropertyRoleClassifier

logger = logging.getLogger(__name__)


# Default mapping from Notion's native property types onto retrieval roles.
# This is the connector-native classification layer (highest-priority automatic
# signal). It is intentionally a plain, overridable mapping rather than
# hardcoded logic so deployments can retune it without code changes.
NOTION_TYPE_ROLE_DEFAULTS: dict[str, PropertyRole] = {
    # Structured, low-cardinality attributes -> filterable retrieval facets.
    "date": PropertyRole.FILTERABLE,
    "checkbox": PropertyRole.FILTERABLE,
    "select": PropertyRole.FILTERABLE,
    "multi_select": PropertyRole.FILTERABLE,
    "status": PropertyRole.FILTERABLE,
    "relation": PropertyRole.FILTERABLE,
    "people": PropertyRole.FILTERABLE,
    "number": PropertyRole.FILTERABLE,
    "url": PropertyRole.FILTERABLE,
    "email": PropertyRole.FILTERABLE,
    "phone_number": PropertyRole.FILTERABLE,
    # Free-form text -> semantic search content.
    "title": PropertyRole.SEMANTIC,
    "rich_text": PropertyRole.SEMANTIC,
    # System/computed fields -> ignored entirely.
    "created_time": PropertyRole.IGNORE,
    "last_edited_time": PropertyRole.IGNORE,
    "created_by": PropertyRole.IGNORE,
    "last_edited_by": PropertyRole.IGNORE,
    "formula": PropertyRole.IGNORE,
    "rollup": PropertyRole.IGNORE,
    "files": PropertyRole.IGNORE,
}


def build_notion_role_classifier(
    config_path: str | Path | None = None,
) -> PropertyRoleClassifier:
    """Build the :class:`PropertyRoleClassifier` used by the Notion connector.

    Starts from :data:`NOTION_TYPE_ROLE_DEFAULTS` and layers optional user
    configuration on top (see :func:`_load_config`):

    - ``type_roles``: override the native-type -> role mapping.
    - ``property_roles``: per-property explicit overrides (highest precedence).

    Missing or malformed configuration is ignored so ingestion always works
    out of the box with sensible defaults.
    """
    type_map: dict[str, PropertyRole] = dict(NOTION_TYPE_ROLE_DEFAULTS)

    path = config_path if config_path is not None else settings.PROPERTY_ROLES_PATH
    config = _load_config(path)

    for raw_type, raw_role in _as_mapping(config.get("type_roles")).items():
        role = PropertyRole.from_value(raw_role)
        if role is not None:
            type_map[str(raw_type).strip().lower()] = role

    overrides = _as_mapping(config.get("property_roles"))

    return PropertyRoleClassifier(type_role_map=type_map, overrides=overrides)


def _load_config(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "Failed to load property roles config from %s", file_path, exc_info=True
        )
        return {}
    return data if isinstance(data, dict) else {}


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
