from app.connectors.base import DocumentConnector
from app.connectors.notion.client import NotionClient
from app.connectors.notion.discovery import (
    ChildPageDiscovery,
    DataSourcePageDiscovery,
    NotionPageDiscovery,
)
from app.connectors.notion.loader import NotionPageLoader
from app.connectors.notion.parser import NotionParser
from app.models.document import Document


class NotionConnector(DocumentConnector):
    """Orchestrates Notion page retrieval into parsed documents."""

    def __init__(
        self,
        client: NotionClient | None = None,
        parser: NotionParser | None = None,
        discoveries: list[NotionPageDiscovery] | None = None,
        loader: NotionPageLoader | None = None,
    ) -> None:
        self.client = client or NotionClient()
        self.parser = parser or NotionParser()
        self.discoveries = (
            discoveries
            if discoveries is not None
            else [
                DataSourcePageDiscovery(self.client),
                ChildPageDiscovery(self.client),
            ]
        )
        self.loader = loader or NotionPageLoader(self.client, self.parser)

    def fetch_documents(self) -> list[Document]:
        self.client.clear_block_cache()
        discovered_pages = []
        visited_page_ids: set[str] = set()

        for discovery in self.discoveries:
            for page in discovery.discover(discovered_pages):
                if page.id in visited_page_ids:
                    continue
                visited_page_ids.add(page.id)
                discovered_pages.append(page)

        documents: list[Document] = []
        for page in discovered_pages:
            document = self.loader.load(page)
            if document is not None:
                documents.append(document)

        return documents
