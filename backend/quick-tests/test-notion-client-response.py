from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.connectors.notion.client import NotionClient

client = NotionClient()

data_sources = client.discover_data_sources()

if not client.health_check():
	print("Notion API is not reachable or the API key is invalid.")
else:
	print(len(data_sources))
	for data_source in data_sources:
		print(f"{data_source.id} :: {data_source.name}")
		pages = client.get_pages(data_source.id)
		print(f"Pages: {len(pages)}")