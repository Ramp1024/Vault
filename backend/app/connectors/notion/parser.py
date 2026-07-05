from typing import Any

from app.connectors.notion.client import NotionDataSource
from models.document import Document


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

		text_parts = [part for part in [title, body] if part]

		return Document(
			id=page_id,
			text="\n\n".join(text_parts),
			metadata={
				"source": "notion",
				"data_source_id": data_source.id,
				"data_source_name": data_source.name,
				"page_title": title,
				"url": page.get("url"),
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

		for block in blocks:
			if not isinstance(block, dict):
				continue

			block_type = block.get("type")
			if not isinstance(block_type, str):
				continue

			block_payload = block.get(block_type)
			if not isinstance(block_payload, dict):
				continue

			rich_text = block_payload.get("rich_text", [])
			line = self._join_rich_text(rich_text)
			if line:
				lines.append(line)

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
