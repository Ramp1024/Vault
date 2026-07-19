import logging

from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService
from app.services.sync_service import SyncService


logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        sync_service = SyncService(
            connector=NotionConnector(),
            chunker=Chunker(),
            embedding_service=EmbeddingService(),
            qdrant_service=QdrantService(client=get_qdrant_client()),
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