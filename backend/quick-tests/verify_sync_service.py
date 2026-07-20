from math import isclose
from pathlib import Path
import sys
from unittest.mock import Mock, call

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.sync_service import SyncService
from app.models.sync_result import SyncResult


def verify_sync_service() -> None:
    connector = Mock()
    chunker = Mock()
    embedding_service = Mock()
    qdrant_service = Mock()
    sync_logger = Mock()
    clock = Mock(side_effect=[100.0, 118.2])

    document = Mock(
        id="document-id",
        metadata={"last_edited_time": "2026-07-19T02:30:00Z"},
    )
    connector.fetch_documents.return_value = [document]
    qdrant_service.collection_exists.return_value = False
    chunker.chunk_documents.return_value = ["chunk"]
    embedding_service.embed_chunks.return_value = ["embedded chunk"]
    qdrant_service.upsert_batch.return_value = 1

    service = SyncService(
        connector=connector,
        chunker=chunker,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
        sync_logger=sync_logger,
        clock=clock,
    )

    result = service.sync()
    assert result == SyncResult(
        documents_processed=1,
        chunks_created=1,
        embeddings_generated=1,
        vectors_upserted=1,
        duration=result.duration,
    )
    assert isclose(result.duration, 18.2)
    connector.fetch_documents.assert_called_once_with()
    chunker.chunk_documents.assert_called_once_with([document])
    embedding_service.embed_chunks.assert_called_once_with(["chunk"])
    qdrant_service.upsert_batch.assert_called_once_with(["embedded chunk"])
    assert sync_logger.info.call_args_list == [
        call("Starting sync..."),
        call("Fetched %d documents", 1),
        call("Skipped %d unchanged documents", 0),
        call("Generated %d chunks", 1),
        call("Embedded %d chunks", 1),
        call("Upserted %d vectors", 1),
        call("Sync completed in %.1f seconds", result.duration),
    ]

    print("SyncService verification passed")


if __name__ == "__main__":
    verify_sync_service()