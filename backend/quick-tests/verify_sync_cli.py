from pathlib import Path
import sys
from unittest.mock import Mock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.cli.sync import main


def verify_sync_cli() -> None:
    connector = Mock()
    chunker = Mock()
    embedding_service = Mock()
    qdrant_client = Mock()
    qdrant_service = Mock()
    sync_service = Mock()

    with (
        patch("app.cli.sync.NotionConnector", return_value=connector),
        patch("app.cli.sync.Chunker", return_value=chunker),
        patch("app.cli.sync.EmbeddingService", return_value=embedding_service),
        patch("app.cli.sync.get_qdrant_client", return_value=qdrant_client),
        patch(
            "app.cli.sync.QdrantService", return_value=qdrant_service
        ) as qdrant_service_class,
        patch("app.cli.sync.SyncService", return_value=sync_service) as sync_service_class,
    ):
        assert main() == 0

        qdrant_service_class.assert_called_once_with(client=qdrant_client)
        sync_service_class.assert_called_once_with(
            connector=connector,
            chunker=chunker,
            embedding_service=embedding_service,
            qdrant_service=qdrant_service,
        )
        sync_service.sync.assert_called_once_with()

        sync_service.sync.side_effect = RuntimeError("Qdrant unavailable")
        with patch("app.cli.sync.logger") as cli_logger:
            assert main() == 1
            cli_logger.error.assert_called_once_with(
                "Sync failed: %s", sync_service.sync.side_effect
            )
            cli_logger.debug.assert_called_once_with(
                "Sync failure details", exc_info=True
            )

        sync_service.sync.side_effect = KeyboardInterrupt()
        with patch("app.cli.sync.logger") as cli_logger:
            assert main() == 130
            cli_logger.error.assert_called_once_with("Sync cancelled.")

    print("Sync CLI verification passed")


if __name__ == "__main__":
    verify_sync_cli()