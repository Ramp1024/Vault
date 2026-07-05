from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.connectors.notion.client import NotionClient

"""
    Temporary test script to check if the Notion API is reachable and the API key is valid.
"""

client = NotionClient()

if client.health_check():
    print("Notion API is reachable and the API key is valid.")
else:
    print("Notion API is not reachable or the API key is invalid.")
