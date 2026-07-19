"""
Story 2 Verification Script

Verifies:
1. Notion authentication
2. Data source discovery
3. Page retrieval
4. Block retrieval
5. Parsing
6. Document generation
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.connectors.notion.client import NotionClient
from app.connectors.notion.connector import NotionConnector


def verify_story2():

    print("=" * 60)
    print("VAULT - STORY 2 VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------
    # Authentication
    # --------------------------------------------------

    client = NotionClient()

    print("\n[1] Authentication")

    if client.health_check():
        print("✅ Connected to Notion")
    else:
        print("❌ Authentication failed")
        return

    # --------------------------------------------------
    # Discover Data Sources
    # --------------------------------------------------

    print("\n[2] Discovering Data Sources")

    data_sources = client.discover_data_sources()

    print(f"Found {len(data_sources)} data sources")

    for source in data_sources:
        print(f"  • {source.name}")

    if len(data_sources) == 0:
        print("❌ No data sources found")
        return

    # --------------------------------------------------
    # Page Retrieval
    # --------------------------------------------------

    print("\n[3] Fetching Pages")

    total_pages = 0

    for source in data_sources:
        pages = client.get_pages(source.id)

        print(f"{source.name}: {len(pages)} pages")

        total_pages += len(pages)

    print(f"\nTotal Pages: {total_pages}")

    if total_pages == 0:
        print("❌ No pages retrieved")
        return

    # --------------------------------------------------
    # End-to-End Connector
    # --------------------------------------------------

    print("\n[4] Loading Documents")

    connector = NotionConnector()

    documents = connector.fetch_documents()

    print(f"Documents Loaded: {len(documents)}")

    if len(documents) == 0:
        print("❌ No documents produced")
        return

    # --------------------------------------------------
    # Sample Output
    # --------------------------------------------------

    sample = documents[0]

    print("\n[5] Sample Document")

    print(f"ID       : {sample.id}")
    print(f"Title    : {sample.title}")
    print(f"Content  : {sample.content[:300]}...")
    print(f"Metadata : {sample.metadata}")

    print("\n" + "=" * 60)
    print("🎉 STORY 2 VERIFIED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    verify_story2()
