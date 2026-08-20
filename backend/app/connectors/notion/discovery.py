from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from app.connectors.notion.client import NotionClient


@dataclass(frozen=True)
class DiscoveredPage:
    id: str
    data_source_id: str | None = None
    data_source_name: str | None = None
    parent_type: str | None = None
    parent_id: str | None = None


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
                            parent_type="data_source_id",
                            parent_id=data_source.id,
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
                    parent_type="page_id",
                    parent_id=parent.id,
                )
                child_pages.append(child_page)
                pending_pages.append(child_page)

        return child_pages


class StandalonePageDiscovery:
    """Discover pages directly via Notion search, independent of data sources.

    Notion search returns every accessible page individually (flattened), so
    standalone workspace pages and their descendants are surfaced here without
    any parent-child traversal. Pages already found through data-source
    traversal are deduplicated by the connector using the Notion page ID.
    """

    def __init__(self, client: NotionClient) -> None:
        self.client = client

    def discover(
        self, discovered_pages: Collection[DiscoveredPage]
    ) -> list[DiscoveredPage]:
        pages: list[DiscoveredPage] = []

        for page in self.client.discover_pages():
            page_id = page.get("id")
            if not isinstance(page_id, str):
                continue

            parent = page.get("parent", {})
            parent_type = parent.get("type") if isinstance(parent, dict) else None
            parent_id = None
            if isinstance(parent, dict) and isinstance(parent_type, str):
                candidate = parent.get(parent_type)
                if isinstance(candidate, str):
                    parent_id = candidate

            pages.append(
                DiscoveredPage(
                    id=page_id,
                    parent_type=parent_type,
                    parent_id=parent_id,
                )
            )

        return pages
