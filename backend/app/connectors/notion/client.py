from dataclasses import dataclass
from typing import Any

from notion_client import Client
from notion_client.errors import APIResponseError

from app.core.config import settings


@dataclass(frozen=True)
class NotionDataSource:
    id: str
    name: str


class NotionClient:
    """
    Thin wrapper around the official Notion SDK

    Responsibilities:
    - Authenticate
    - Expose underlying SDK client
    - Verify connectivity

    This class should not:
    - Parse pages
    - Build Document objects
    - Perform indexing
    """

    def __init__(self):
        self._client = (
            Client(auth=settings.NOTION_API_KEY) if settings.NOTION_API_KEY else None
        )

    @property
    def client(self) -> Client | None:
        return self._client

    def health_check(self) -> bool:
        """
        Check if the Notion API is reachable and the API key is valid.
        Returns True if the API is reachable and the API key is valid, False otherwise.
        """
        if self.client is None:
            return False

        try:
            self.client.search(
                page_size=1
            )  # Perform a simple search to check connectivity
            return True
        except APIResponseError:
            # Log the error if needed
            return False

    def discover_data_sources(self) -> list[NotionDataSource]:
        """Discover all data sources accessible to the integration."""
        if self.client is None:
            return []

        try:
            response = self.client.search(
                page_size=100,
                filter={"property": "object", "value": "data_source"},
            )
            results = response.get("results", [])
            data_sources: list[NotionDataSource] = []

            for result in results:
                data_source_id = result.get("id")
                title = self._extract_title(result)
                if data_source_id:
                    data_sources.append(NotionDataSource(id=data_source_id, name=title))

            return data_sources
        except APIResponseError:
            # Log the error if needed
            return []

    def get_pages(self, data_source_id: str) -> list[dict[str, Any]]:
        """Return all pages for a given Notion data source."""
        if self.client is None:
            return []

        try:
            response = self.client.data_sources.query(data_source_id=data_source_id)
            return response.get("results", [])
        except APIResponseError:
            return []

    def get_page_blocks(self, page_id: str) -> list[dict[str, Any]]:
        """Return top-level blocks for a page."""
        if self.client is None:
            return []

        try:
            response = self.client.blocks.children.list(block_id=page_id)
            return response.get("results", [])
        except APIResponseError:
            return []

    @staticmethod
    def _extract_title(result: dict[str, Any]) -> str:
        title = result.get("title", [])
        if isinstance(title, list) and title:
            first = title[0]
            if isinstance(first, dict):
                text = first.get("plain_text") or first.get("text", {}).get("content")
                if isinstance(text, str) and text:
                    return text

        return "Untitled Data Source"
