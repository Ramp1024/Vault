import logging

from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService
from app.services.sync_service import SyncService


logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        qdrant_service = QdrantService(client=get_qdrant_client())
        sync_service = SyncService(
            connector=NotionConnector(),
            chunker=Chunker(),
            embedding_service=EmbeddingService(),
            qdrant_service=qdrant_service,
            # Rebuild the lexical index after each sync so hybrid retrieval stays
            # in step with the vector store instead of going stale.
            bm25_indexer=BM25Indexer(qdrant_service=qdrant_service),
        )
        sync_service.sync()
    except KeyboardInterrupt:
        logger.error("Sync cancelled.")
        return 130
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        logger.debug("Sync failure details", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())