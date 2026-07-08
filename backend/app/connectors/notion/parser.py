from typing import Any

from app.connectors.notion.client import NotionDataSource
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

        return Document(
            id=page_id,
            title=title,
            content=body,
            metadata={
                "source": "notion",
                "data_source_id": data_source.id,
                "data_source_name": data_source.name,
                "url": page.get("url"),
                "properties": self._format_properties(properties),
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

    def _format_properties(self, properties: dict[str, Any]) -> str:
        """Extract and format Notion page properties into a readable string."""
        if not isinstance(properties, dict) or not properties:
            return ""

        formatted_props: list[str] = []

        for prop_name, prop_data in properties.items():
            if not isinstance(prop_data, dict):
                continue

            prop_type = prop_data.get("type")

            # Extract value based on property type
            if prop_type == "checkbox":
                value = prop_data.get("checkbox", False)
                formatted_props.append(f"{prop_name}: {value}")
            elif prop_type == "select":
                select_obj = prop_data.get("select")
                if select_obj and isinstance(select_obj, dict):
                    formatted_props.append(f"{prop_name}: {select_obj.get('name', '')}")
            elif prop_type == "multi_select":
                multi_select = prop_data.get("multi_select", [])
                if isinstance(multi_select, list):
                    tags = ", ".join(
                        [
                            item.get("name", "")
                            for item in multi_select
                            if isinstance(item, dict)
                        ]
                    )
                    if tags:
                        formatted_props.append(f"{prop_name}: {tags}")
            elif prop_type == "date":
                date_obj = prop_data.get("date")
                if date_obj and isinstance(date_obj, dict):
                    start_date = date_obj.get("start", "")
                    end_date = date_obj.get("end")
                    if end_date:
                        formatted_props.append(
                            f"{prop_name}: {start_date} → {end_date}"
                        )
                    else:
                        formatted_props.append(f"{prop_name}: {start_date}")
            elif prop_type == "rich_text":
                rich_text = prop_data.get("rich_text", [])
                if rich_text:
                    text = self._join_rich_text(rich_text)
                    if text:
                        formatted_props.append(f"{prop_name}: {text}")
            elif prop_type == "number":
                number_val = prop_data.get("number")
                if number_val is not None:
                    formatted_props.append(f"{prop_name}: {number_val}")
            elif prop_type == "url":
                url_val = prop_data.get("url")
                if url_val:
                    formatted_props.append(f"{prop_name}: {url_val}")

        return " | ".join(formatted_props)

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
