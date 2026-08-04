from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

from app.core.config import settings
from app.models.metadata_schema import FieldType, MetadataSchema

logger = logging.getLogger(__name__)


class PayloadIndexManager:
    """Provision Qdrant payload indexes from a :class:`MetadataSchema`.

    The schema is the single source of truth for which metadata fields are
    filterable, so it also drives which payload indexes Qdrant needs. This keeps
    index provisioning connector-agnostic: any connector that produces a schema
    automatically gets the right indexes, with no hardcoded field names.

    Fields are stored under a nested ``properties`` object, so a logical field
    ``date`` is indexed at the payload path ``properties.date`` — matching the
    key layout used by :class:`app.services.qdrant_filter_builder.QdrantFilterBuilder`.

    Provisioning is idempotent: existing indexes with the expected type are left
    untouched, so it is safe to run on every sync or startup.
    """

    PROPERTY_PREFIX = "properties"

    # Map each logical field type onto the Qdrant payload index that makes its
    # operators executable server-side.
    #   DATE    -> DATETIME: enables server-side date range/equality filters.
    #   NUMBER  -> FLOAT:    enables numeric range/equality filters.
    #   BOOLEAN -> BOOL:     enables exact boolean matching.
    #   STRING  -> KEYWORD:  enables exact match / MatchAny membership.
    _TYPE_MAP: dict[FieldType, PayloadSchemaType] = {
        FieldType.DATE: PayloadSchemaType.DATETIME,
        FieldType.NUMBER: PayloadSchemaType.FLOAT,
        FieldType.BOOLEAN: PayloadSchemaType.BOOL,
        FieldType.STRING: PayloadSchemaType.KEYWORD,
    }

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str | None = None,
        index_logger: logging.Logger | None = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.logger = index_logger if index_logger is not None else logger

    def ensure_indexes(self, schema: MetadataSchema) -> int:
        """Ensure a payload index exists for every filterable field in ``schema``.

        Returns the number of indexes created. Missing collection or empty
        schema is a no-op. Individual failures are logged and skipped so index
        provisioning never blocks a sync.
        """
        if not schema:
            return 0
        if not self.client.collection_exists(collection_name=self.collection_name):
            return 0

        existing = self._existing_indexes()
        created = 0
        for field in schema.fields:
            schema_type = self._TYPE_MAP.get(field.type)
            if schema_type is None:
                continue
            key = self._payload_key(field.name)
            if existing.get(key) == schema_type:
                # Already indexed with the expected type; nothing to do.
                continue
            if self._create_index(key, schema_type):
                created += 1
        return created

    def _payload_key(self, field_name: str) -> str:
        return f"{self.PROPERTY_PREFIX}.{field_name}"

    def _create_index(self, key: str, schema_type: PayloadSchemaType) -> bool:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=key,
                field_schema=schema_type,
                wait=True,
            )
            self.logger.info("Ensured payload index %s (%s)", key, schema_type.value)
            return True
        except Exception:
            self.logger.warning(
                "Failed to create payload index for %s (%s)",
                key,
                schema_type.value,
                exc_info=True,
            )
            return False

    def _existing_indexes(self) -> dict[str, PayloadSchemaType]:
        """Return a map of already-indexed payload keys to their index type."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.logger.warning(
                "Failed to read existing payload indexes", exc_info=True
            )
            return {}

        payload_schema = getattr(info, "payload_schema", None) or {}
        existing: dict[str, PayloadSchemaType] = {}
        for name, index_info in payload_schema.items():
            data_type = getattr(index_info, "data_type", None)
            if data_type is not None:
                existing[name] = data_type
        return existing
