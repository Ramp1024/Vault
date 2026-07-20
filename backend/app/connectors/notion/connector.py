from app.connectors.base import DocumentConnector
from app.connectors.notion.client import NotionClient
from app.connectors.notion.parser import NotionParser
from app.models.document import Document


class NotionConnector(DocumentConnector):
    """Orchestrates Notion page retrieval into parsed documents."""

    def __init__(
        self, client: NotionClient | None = None, parser: NotionParser | None = None
    ):
        self.client = client or NotionClient()
        self.parser = parser or NotionParser()

    def fetch_documents(self) -> list[Document]:
        documents: list[Document] = []
        data_sources = self.client.discover_data_sources()
        for data_source in data_sources:
            pages = self.client.get_pages(data_source.id)
            for page in pages:
                page_id = page.get("id")
                if not page_id:
                    continue

                blocks = self.client.get_page_blocks(page_id)
                document = self.parser.parse_page(data_source, page, blocks)
                documents.append(document)

        return documents
