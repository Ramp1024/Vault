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
        self._block_children_cache: dict[str, list[dict[str, Any]]] = {}

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
            data_sources: list[NotionDataSource] = []
            start_cursor: str | None = None

            while True:
                response = self.client.search(
                    page_size=100,
                    start_cursor=start_cursor,
                    filter={"property": "object", "value": "data_source"},
                )

                for result in response.get("results", []):
                    data_source_id = result.get("id")
                    title = self._extract_title(result)
                    if data_source_id:
                        data_sources.append(
                            NotionDataSource(id=data_source_id, name=title)
                        )

                if not response.get("has_more", False):
                    break

                next_cursor = response.get("next_cursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                start_cursor = next_cursor

            return data_sources
        except APIResponseError:
            # Log the error if needed
            return []

    def get_pages(self, data_source_id: str) -> list[dict[str, Any]]:
        """Return all pages for a given Notion data source."""
        if self.client is None:
            return []

        try:
            pages: list[dict[str, Any]] = []
            start_cursor: str | None = None

            while True:
                response = self.client.data_sources.query(
                    data_source_id=data_source_id,
                    page_size=100,
                    start_cursor=start_cursor,
                )
                pages.extend(response.get("results", []))

                if not response.get("has_more", False):
                    break

                next_cursor = response.get("next_cursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                start_cursor = next_cursor

            return pages
        except APIResponseError:
            return []

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        """Return one page by ID."""
        if self.client is None:
            return None

        try:
            return self.client.pages.retrieve(page_id=page_id)
        except APIResponseError:
            return None

    def get_block_children(self, block_id: str) -> list[dict[str, Any]]:
        """Return all direct children of a block using cursor pagination."""
        if self.client is None:
            return []
        if block_id in self._block_children_cache:
            return self._block_children_cache[block_id]

        try:
            children: list[dict[str, Any]] = []
            start_cursor: str | None = None

            while True:
                response = self.client.blocks.children.list(
                    block_id=block_id,
                    page_size=100,
                    start_cursor=start_cursor,
                )
                children.extend(response.get("results", []))

                if not response.get("has_more", False):
                    break

                next_cursor = response.get("next_cursor")
                if not isinstance(next_cursor, str) or not next_cursor:
                    break
                start_cursor = next_cursor

            self._block_children_cache[block_id] = children
            return children
        except APIResponseError:
            return []

    def clear_block_cache(self) -> None:
        """Clear block responses cached during the current connector run."""
        self._block_children_cache.clear()

    def get_child_page_ids(self, page_id: str) -> list[str]:
        """Discover child-page references nested anywhere in a page's blocks."""
        child_page_ids: list[str] = []
        pending_block_ids = [page_id]
        visited_block_ids: set[str] = set()

        while pending_block_ids:
            block_id = pending_block_ids.pop()
            if block_id in visited_block_ids:
                continue
            visited_block_ids.add(block_id)

            for block in self.get_block_children(block_id):
                child_block_id = block.get("id")
                if not isinstance(child_block_id, str):
                    continue

                if block.get("type") == "child_page":
                    child_page_ids.append(child_block_id)
                elif block.get("has_children"):
                    pending_block_ids.append(child_block_id)

        return child_page_ids

    def get_page_blocks(self, page_id: str) -> list[dict[str, Any]]:
        """Return all blocks for a page, recursively fetching child blocks and table rows.
        
        This method:
        1. Fetches top-level blocks from the page
        2. For blocks with children (except child_page), recursively fetches those blocks
        3. For tables, extracts row content from child blocks
        4. Flattens the structure so child blocks are included in the main list
        5. Child pages are NOT traversed - they are separate documents
        """
        return self._get_blocks_recursive(page_id)

    def _get_blocks_recursive(self, block_id: str) -> list[dict[str, Any]]:
        """Recursively fetch all child blocks for a given block ID.
        
        Note: child_page blocks are NOT traversed as they are separate documents.
        """
        all_blocks: list[dict[str, Any]] = []

        for block in self.get_block_children(block_id):
            block_type = block.get("type")
            if block_type == "child_page":
                continue

            all_blocks.append(block)

            child_block_id = block.get("id")
            if not block.get("has_children") or not isinstance(child_block_id, str):
                continue

            child_blocks = self._get_blocks_recursive(child_block_id)
            if block_type == "table":
                table_rows = self._extract_table_rows(child_blocks)
                if table_rows:
                    block["table_rows"] = table_rows
            elif child_blocks:
                block["children"] = child_blocks
                all_blocks.extend(child_blocks)

        return all_blocks

    def _extract_table_rows(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract table row content from a list of blocks.
        
        Notion represents table rows as child blocks of type 'table_row'.
        Each table_row contains cells with rich_text.
        """
        rows: list[dict[str, Any]] = []
        
        for block in blocks:
            if block.get("type") == "table_row":
                row_data = {
                    "type": "table_row",
                    "id": block.get("id"),
                    "cells": [],
                }
                
                # Extract cell content
                row_payload = block.get("table_row", {})
                cells = row_payload.get("cells", [])
                
                for cell in cells:
                    # Each cell is a list of rich text objects
                    cell_text = self._extract_rich_text_from_cell(cell)
                    row_data["cells"].append(cell_text)
                
                rows.append(row_data)
        
        return rows

    @staticmethod
    def _extract_rich_text_from_cell(cell: list[dict[str, Any]]) -> str:
        """Extract plain text from a table cell (which is a list of rich_text objects)."""
        if not isinstance(cell, list):
            return ""
        
        parts: list[str] = []
        for item in cell:
            if isinstance(item, dict):
                plain_text = item.get("plain_text", "")
                if plain_text:
                    parts.append(plain_text)
        
        return "".join(parts).strip()

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
