from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException

from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService
from app.services.sync_service import SyncService


logger = logging.getLogger(__name__)

router = APIRouter()

# Serialize sync runs so a second click can't kick off a concurrent, competing
# rebuild of the vector and lexical indexes.
_sync_lock = threading.Lock()


def _run_sync() -> dict:
    qdrant_service = QdrantService(client=get_qdrant_client())
    sync_service = SyncService(
        connector=NotionConnector(),
        chunker=Chunker(),
        embedding_service=EmbeddingService(),
        qdrant_service=qdrant_service,
        bm25_indexer=BM25Indexer(qdrant_service=qdrant_service),
    )
    result = sync_service.sync()
    return {
        "documents_processed": result.documents_processed,
        "documents_skipped": result.documents_skipped,
        "chunks_created": result.chunks_created,
        "embeddings_generated": result.embeddings_generated,
        "vectors_upserted": result.vectors_upserted,
        "duration": round(result.duration, 1),
    }


@router.post("/sync")
def trigger_sync() -> dict:
    """Trigger a full sync of source documents into the vector store.

    Defined as a sync function so FastAPI runs it in a worker thread, keeping the
    event loop responsive during the long-running sync.
    """
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A sync is already in progress.")
    try:
        return _run_sync()
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        logger.debug("Sync failure details", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sync failed: {exc}") from exc
    finally:
        _sync_lock.release()
