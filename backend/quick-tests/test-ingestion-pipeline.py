from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.connectors.notion.connector import NotionConnector

connector = NotionConnector()
documents = connector.ingest()

print(len(documents))
if documents:
	print(documents[0].text)
