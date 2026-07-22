from typing import Any

from app.connectors.notion.client import NotionDataSource
from app.core.text import to_camel_case
from app.models.document import Document


class NotionParser:
    """Parse raw Notion API payloads into domain Documents."""

    def parse_page(
        self,
        data_source: NotionDataSource,
        page: dict[str, Any],
        blocks: list[dict[str, Any]],
    ) -> Document:
        page_id = str(page.get("id", ""))
        title = self._extract_page_title(page)
        body = self._extract_blocks_text(blocks)
        properties = page.get("properties", {})
        last_edited_time = page.get("last_edited_time", "")
        if not isinstance(last_edited_time, str):
            last_edited_time = ""

        return Document(
            id=page_id,
            title=title,
            content=body,
            metadata={
                "source": "notion",
                "data_source_id": data_source.id,
                "data_source_name": data_source.name,
                "last_edited_time": last_edited_time,
                "url": page.get("url"),
                "properties": self._extract_properties(properties),
            },
        )

    def _extract_page_title(self, page: dict[str, Any]) -> str:
        properties = page.get("properties", {})
        if not isinstance(properties, dict):
            return ""

        for prop in properties.values():
            if not isinstance(prop, dict):
                continue
            if prop.get("type") != "title":
                continue

            title_items = prop.get("title", [])
            return self._join_rich_text(title_items)

        return ""

    def _extract_blocks_text(self, blocks: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        list_index = 0

        for block in blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if not isinstance(block_type, str):
                continue

            block_payload = block.get(block_type)
            if not isinstance(block_payload, dict):
                continue

            # Handle table blocks with rows
            if block_type == "table":
                table_rows = block.get("table_rows", [])
                if table_rows:
                    table_lines = self._format_table(table_rows)
                    lines.extend(table_lines)
                continue

            # Handle callout blocks (treat like quote but with icon)
            if block_type == "callout":
                rich_text = block_payload.get("rich_text", [])
                text = self._join_rich_text(rich_text)
                if text:
                    lines.append(f"💡 {text}")
                continue

            rich_text = block_payload.get("rich_text", [])
            text = self._join_rich_text(rich_text)

            if not text:
                continue

            # Format based on block type
            if block_type == "heading_1":
                lines.append(f"# {text}")
            elif block_type == "heading_2":
                lines.append(f"## {text}")
            elif block_type == "heading_3":
                lines.append(f"### {text}")
            elif block_type == "bulleted_list_item":
                lines.append(f"- {text}")
            elif block_type == "numbered_list_item":
                list_index += 1
                lines.append(f"{list_index}. {text}")
            elif block_type == "quote":
                lines.append(f"> {text}")
            elif block_type == "code":
                lines.append(f"```\n{text}\n```")
            elif block_type == "paragraph":
                lines.append(text)
            else:
                # Default for other block types
                lines.append(text)

        return "\n".join(lines)

    def _join_rich_text(self, rich_text: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in rich_text:
            if not isinstance(item, dict):
                continue
            plain_text = item.get("plain_text")
            if isinstance(plain_text, str):
                parts.append(plain_text)

        return "".join(parts).strip()

    def _extract_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Extract Notion page properties into a typed field/value mapping.

        Keys are normalized to camelCase (e.g. ``"Leetcode Topic"`` ->
        ``"leetcodeTopic"``) for consistency across connectors; values are
        normalized to plain Python types (str, int/float, bool, list[str]) so
        they can be stored as individual payload fields and used for metadata
        filtering. The page title property is skipped since it is captured
        separately.
        """
        if not isinstance(properties, dict) or not properties:
            return {}

        extracted: dict[str, Any] = {}

        for prop_name, prop_data in properties.items():
            if not isinstance(prop_data, dict):
                continue

            prop_type = prop_data.get("type")
            key = to_camel_case(prop_name)
            if not key:
                continue

            if prop_type == "checkbox":
                extracted[key] = bool(prop_data.get("checkbox", False))
            elif prop_type == "select":
                select_obj = prop_data.get("select")
                if isinstance(select_obj, dict):
                    name = select_obj.get("name")
                    if name:
                        extracted[key] = name
            elif prop_type == "multi_select":
                multi_select = prop_data.get("multi_select", [])
                if isinstance(multi_select, list):
                    tags = [
                        item.get("name", "")
                        for item in multi_select
                        if isinstance(item, dict) and item.get("name")
                    ]
                    if tags:
                        extracted[key] = tags
            elif prop_type == "date":
                date_obj = prop_data.get("date")
                if isinstance(date_obj, dict):
                    start_date = date_obj.get("start")
                    if start_date:
                        extracted[key] = start_date
            elif prop_type == "rich_text":
                rich_text = prop_data.get("rich_text", [])
                if rich_text:
                    text = self._join_rich_text(rich_text)
                    if text:
                        extracted[key] = text
            elif prop_type == "number":
                number_val = prop_data.get("number")
                if number_val is not None:
                    extracted[key] = number_val
            elif prop_type == "url":
                url_val = prop_data.get("url")
                if url_val:
                    extracted[key] = url_val

        return extracted

    def _format_table(self, table_rows: list[dict[str, Any]]) -> list[str]:
        """Format table rows into markdown pipe table format.
        
        Args:
            table_rows: List of table_row objects with cells
            
        Returns:
            List of formatted table lines
        """
        if not table_rows:
            return []

        lines: list[str] = []

        for row in table_rows:
            cells = row.get("cells", [])
            if cells:
                # Join cells with pipe separator
                row_text = " | ".join(cells)
                lines.append(f"| {row_text} |")

        return lines
