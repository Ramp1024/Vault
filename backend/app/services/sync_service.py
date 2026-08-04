import logging
from collections.abc import Callable
from time import perf_counter

from app.connectors.base import DocumentConnector
from app.models.sync_result import SyncResult
from app.processors.chunker import Chunker
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.metadata_schema_store import MetadataSchemaStore
from app.services.qdrant_service import QdrantService


logger = logging.getLogger(__name__)


class SyncService:
    """Synchronize source documents into the vector store."""

    def __init__(
        self,
        connector: DocumentConnector,
        chunker: Chunker,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        sync_logger: logging.Logger | None = None,
        clock: Callable[[], float] = perf_counter,
        bm25_indexer: BM25Indexer | None = None,
        schema_store: MetadataSchemaStore | None = None,
    ) -> None:
        self.connector = connector
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.logger = sync_logger if sync_logger is not None else logger
        self.clock = clock
        self.bm25_indexer = bm25_indexer
        self.schema_store = (
            schema_store if schema_store is not None else MetadataSchemaStore()
        )

    def sync(self) -> SyncResult:
        started_at = self.clock()
        self.logger.info("Starting sync...")

        documents = self.connector.fetch_documents()
        self.logger.info("Fetched %d documents", len(documents))

        self._persist_schema(documents)

        documents_to_index = []
        collection_exists = self.qdrant_service.collection_exists()

        for document in documents:
            last_edited_time = document.metadata.get("last_edited_time")
            indexed_metadata = (
                self.qdrant_service.get_document_metadata(document.id)
                if collection_exists
                else None
            )
            if indexed_metadata is None:
                documents_to_index.append(document)
                continue

            if (
                isinstance(last_edited_time, str)
                and last_edited_time
                and indexed_metadata.get("last_edited_time")
                == last_edited_time
            ):
                continue

            self.qdrant_service.delete_document(document.id)
            documents_to_index.append(document)

        documents_skipped = len(documents) - len(documents_to_index)
        self.logger.info("Skipped %d unchanged documents", documents_skipped)

        chunks = self.chunker.chunk_documents(documents_to_index)
        self.logger.info("Generated %d chunks", len(chunks))

        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        self.logger.info("Embedded %d chunks", len(embedded_chunks))

        vectors_upserted = self.qdrant_service.upsert_batch(embedded_chunks)
        self.logger.info("Upserted %d vectors", vectors_upserted)

        if self.bm25_indexer is not None:
            # Rebuild the lexical index from the full vector store so it always
            # references the same logical chunks as the vector index.
            bm25_indexed = self.bm25_indexer.rebuild()
            self.logger.info("Rebuilt BM25 index over %d chunks", bm25_indexed)

        duration = self.clock() - started_at
        result = SyncResult(
            documents_processed=len(documents_to_index),
            chunks_created=len(chunks),
            embeddings_generated=len(embedded_chunks),
            vectors_upserted=vectors_upserted,
            duration=duration,
            documents_skipped=documents_skipped,
        )
        self.logger.info("Sync completed in %.1f seconds", duration)
        return result

    def _persist_schema(self, documents: list) -> None:
        """Discover and persist the filterable metadata schema for this sync.

        The connector infers its schema from the freshly fetched documents (no
        extra fetch), and the result is written to the schema store so the
        schema-aware intent analyzer can load it at query time. Discovery never
        blocks a sync: any failure is logged and ignored.
        """
        try:
            schema = self.connector.describe_schema(documents)
            self.schema_store.save(schema)
            self.logger.info(
                "Persisted metadata schema with %d fields", len(schema.fields)
            )
        except Exception:
            self.logger.warning("Failed to persist metadata schema", exc_info=True)
