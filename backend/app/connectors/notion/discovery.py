from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from app.connectors.notion.client import NotionClient


@dataclass(frozen=True)
class DiscoveredPage:
    id: str
    data_source_id: str
    data_source_name: str


# Any object with a compatible discover() method can be used as a Notion discovery strategy.
class NotionPageDiscovery(Protocol):
    def discover(
        self, discovered_pages: Collection[DiscoveredPage]
    ) -> list[DiscoveredPage]: ...


class DataSourcePageDiscovery:
    """Discover pages that are direct rows of accessible data sources."""

    def __init__(self, client: NotionClient) -> None:
        self.client = client

    def discover(
        self, discovered_pages: Collection[DiscoveredPage]
    ) -> list[DiscoveredPage]:
        pages: list[DiscoveredPage] = []

        for data_source in self.client.discover_data_sources():
            for page in self.client.get_pages(data_source.id):
                page_id = page.get("id")
                if isinstance(page_id, str):
                    pages.append(
                        DiscoveredPage(
                            id=page_id,
                            data_source_id=data_source.id,
                            data_source_name=data_source.name,
                        )
                    )

        return pages


class ChildPageDiscovery:
    """Recursively discover child pages beneath already discovered pages."""

    def __init__(self, client: NotionClient) -> None:
        self.client = client

    def discover(
        self, discovered_pages: Collection[DiscoveredPage]
    ) -> list[DiscoveredPage]:
        known_page_ids = {page.id for page in discovered_pages}
        pending_pages = list(discovered_pages)
        child_pages: list[DiscoveredPage] = []

        while pending_pages:
            parent = pending_pages.pop()
            for child_page_id in self.client.get_child_page_ids(parent.id):
                if child_page_id in known_page_ids:
                    continue

                known_page_ids.add(child_page_id)
                child_page = DiscoveredPage(
                    id=child_page_id,
                    data_source_id=parent.data_source_id,
                    data_source_name=parent.data_source_name,
                )
                child_pages.append(child_page)
                pending_pages.append(child_page)

        return child_pages
