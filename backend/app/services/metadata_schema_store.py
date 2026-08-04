from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import settings
from app.models.metadata_schema import MetadataSchema

logger = logging.getLogger(__name__)


class MetadataSchemaStore:
    """Persist and load the discovered :class:`MetadataSchema` as JSON.

    Ingestion writes the schema so query-time components (the LLM intent
    analyzer) can load the exact fields, types, and operators the connector
    produced, without re-running discovery or contacting the source. The format
    is plain JSON so the schema is easy to inspect and review.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path(settings.METADATA_SCHEMA_PATH)

    def save(self, schema: MetadataSchema) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(schema.to_dict(), indent=2, ensure_ascii=False)
        self.path.write_text(payload + "\n", encoding="utf-8")

    def load(self) -> MetadataSchema | None:
        """Load the persisted schema, or None if it is missing/unreadable."""
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to read metadata schema at %s", self.path)
            return None
        return MetadataSchema.from_dict(raw)

    def exists(self) -> bool:
        return self.path.exists()
