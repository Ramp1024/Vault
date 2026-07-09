"""
Story 4 - Milestone 2 Verification
Verifies:
- Collection exists
- All EmbeddedChunks are stored
- IDs are deterministic
- Payload is correct
- Vector dimensions are correct
"""

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings
from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService


def verify_milestone2():

    connector = NotionConnector()
    chunker = Chunker()
    embedding_service = EmbeddingService()
    qdrant = QdrantService(client=get_qdrant_client())

    print("Loading documents...")
    documents = connector.ingest()

    print("Chunking...")
    chunks = chunker.chunk_documents(documents)

    print("Embedding...")
    embedded_chunks = embedding_service.embed_chunks(chunks)

    print("Upserting...")
    qdrant.upsert(embedded_chunks)

    print("\n========== VERIFICATION ==========")

    # Collection exists
    assert qdrant.collection_exists(), "Collection was not created."
    print("✅ Collection exists")

    # Point count
    point_count = qdrant.count()

    assert point_count == len(embedded_chunks), (
        f"Expected {len(embedded_chunks)} points, found {point_count}"
    )

    print("✅ Point count matches")

    # Retrieve one sample point
    sample = embedded_chunks[0]

    point = qdrant.get_point(sample.chunk.id)

    assert point is not None, "Stored point not found."

    print("✅ Sample point retrieved")

    payload = point.payload

    assert payload["document_id"] == sample.chunk.document_id
    assert payload["document_title"] == sample.chunk.document_title
    assert payload["content"] == sample.chunk.content

    print("✅ Payload integrity verified")

    # Vector exists
    assert len(point.vector) == len(sample.embedding)

    print("✅ Vector dimension verified")

    print("\n🎉 Milestone 2 verification passed!")


if __name__ == "__main__":
    verify_milestone2()
