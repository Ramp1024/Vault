from app.connectors.notion.client import NotionClient, NotionDataSource
from app.connectors.notion.discovery import DiscoveredPage
from app.connectors.notion.parser import NotionParser
from app.models.document import Document


class NotionPageLoader:
    """Load one discovered Notion page into one domain Document."""

    def __init__(self, client: NotionClient, parser: NotionParser) -> None:
        self.client = client
        self.parser = parser

    def load(self, discovered_page: DiscoveredPage) -> Document | None:
        page = self.client.get_page(discovered_page.id)
        if page is None:
            return None

        blocks = self.client.get_page_blocks(discovered_page.id)
        data_source = NotionDataSource(
            id=discovered_page.data_source_id,
            name=discovered_page.data_source_name,
        )
        return self.parser.parse_page(data_source, page, blocks)