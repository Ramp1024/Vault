from typing import Any

from app.connectors.notion.client import NotionDataSource
from app.connectors.notion.property_roles import build_notion_role_classifier
from app.core.text import to_camel_case
from app.models.document import SEMANTIC_PROPERTIES_KEY, Document
from app.models.property_role import PropertyRole
from app.processors.property_role_classifier import PropertyRoleClassifier


class NotionParser:
    """Parse raw Notion API payloads into domain Documents."""

    def __init__(self, role_classifier: PropertyRoleClassifier | None = None) -> None:
        self.role_classifier = (
            role_classifier
            if role_classifier is not None
            else build_notion_role_classifier()
        )

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

        filterable, semantic = self._classify_properties(properties)

        metadata: dict[str, Any] = {
            "source": "notion",
            "data_source_id": data_source.id,
            "data_source_name": data_source.name,
            "last_edited_time": last_edited_time,
            "url": page.get("url"),
            "properties": filterable,
        }
        if semantic:
            metadata[SEMANTIC_PROPERTIES_KEY] = semantic

        return Document(
            id=page_id,
            title=title,
            content=body,
            metadata=metadata,
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

    def _classify_properties(
        self, properties: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Split Notion page properties into filterable and semantic buckets.

        Each property is first assigned a :class:`PropertyRole` (independently of
        its datatype) and then routed accordingly:

        - ``FILTERABLE`` properties are extracted into a typed field/value
          mapping keyed by camelCase name (e.g. ``"Leetcode Topic"`` ->
          ``"leetcodeTopic"``). These drive metadata schema discovery, payload
          indexes, and filtering.
        - ``SEMANTIC`` properties are extracted as plain text keyed by their
          original name so the chunker can merge them into the page body.
        - ``IGNORE`` properties are discarded.

        The title property is always skipped here since it is captured
        separately as the document title.
        """
        if not isinstance(properties, dict) or not properties:
            return {}, {}

        filterable: dict[str, Any] = {}
        semantic: dict[str, str] = {}

        for prop_name, prop_data in properties.items():
            if not isinstance(prop_data, dict):
                continue

            prop_type = prop_data.get("type")
            if prop_type == "title":
                continue

            key = to_camel_case(prop_name)
            if not key:
                continue

            value = self._extract_value(prop_type, prop_data)
            role = self.role_classifier.classify(
                name=prop_name,
                key=key,
                native_type=prop_type,
                value=value,
            )

            if role is PropertyRole.IGNORE:
                continue
            if value is None:
                continue

            if role is PropertyRole.FILTERABLE:
                filterable[key] = value
            else:  # PropertyRole.SEMANTIC
                text = self._value_as_text(value)
                if text:
                    semantic[prop_name] = text

        return filterable, semantic

    def _extract_value(self, prop_type: Any, prop_data: dict[str, Any]) -> Any:
        """Normalize a single Notion property into a plain Python value.

        Returns ``None`` when the property has no extractable value (empty or an
        unsupported type). The native type is preserved by the caller for role
        classification; this method only normalizes the value.
        """
        if prop_type == "checkbox":
            return bool(prop_data.get("checkbox", False))
        if prop_type == "select":
            select_obj = prop_data.get("select")
            if isinstance(select_obj, dict):
                name = select_obj.get("name")
                if name:
                    return name
            return None
        if prop_type == "status":
            status_obj = prop_data.get("status")
            if isinstance(status_obj, dict):
                name = status_obj.get("name")
                if name:
                    return name
            return None
        if prop_type == "multi_select":
            multi_select = prop_data.get("multi_select", [])
            if isinstance(multi_select, list):
                tags = [
                    item.get("name", "")
                    for item in multi_select
                    if isinstance(item, dict) and item.get("name")
                ]
                if tags:
                    return tags
            return None
        if prop_type == "people":
            people = prop_data.get("people", [])
            if isinstance(people, list):
                names = [
                    person.get("name", "")
                    for person in people
                    if isinstance(person, dict) and person.get("name")
                ]
                if names:
                    return names
            return None
        if prop_type == "relation":
            relations = prop_data.get("relation", [])
            if isinstance(relations, list):
                ids = [
                    item.get("id", "")
                    for item in relations
                    if isinstance(item, dict) and item.get("id")
                ]
                if ids:
                    return ids
            return None
        if prop_type == "date":
            date_obj = prop_data.get("date")
            if isinstance(date_obj, dict):
                start_date = date_obj.get("start")
                if start_date:
                    return start_date
            return None
        if prop_type == "rich_text":
            text = self._join_rich_text(prop_data.get("rich_text", []))
            return text or None
        if prop_type == "number":
            return prop_data.get("number")
        if prop_type == "url":
            return prop_data.get("url") or None
        if prop_type == "email":
            return prop_data.get("email") or None
        if prop_type == "phone_number":
            return prop_data.get("phone_number") or None
        return None

    @staticmethod
    def _value_as_text(value: Any) -> str:
        """Render a semantic property value as plain text for embedding."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        if value is None:
            return ""
        return str(value).strip()

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
