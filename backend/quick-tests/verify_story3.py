"""
Story 3 Verification

Verifies the complete pipeline:

Notion -> Documents -> Chunks -> Embeddings
"""

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker
from app.services.embedding_service import get_embedding_service


def verify_story3():

    print("=" * 70)
    print("VAULT - STORY 3 VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load Documents
    # --------------------------------------------------

    connector = NotionConnector()
    documents = connector.ingest()

    assert len(documents) > 0, "No documents loaded."

    print(f"✅ Documents Loaded      : {len(documents)}")

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0, "No chunks generated."

    print(f"✅ Chunks Generated      : {len(chunks)}")

    # Every document should produce at least one chunk
    document_ids = {d.id for d in documents}
    chunk_document_ids = {c.document_id for c in chunks}

    assert document_ids == chunk_document_ids, "Some documents were not chunked."

    # Validate chunk IDs
    seen_ids = set()

    for chunk in chunks:
        assert chunk.id not in seen_ids, f"Duplicate chunk id: {chunk.id}"

        seen_ids.add(chunk.id)

        assert chunk.content.strip(), f"Empty content in {chunk.id}"

        assert chunk.document_title in chunk.content, (
            f"Missing document title in {chunk.id}"
        )

    print("✅ Chunk Validation      : Passed")

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    embedding_service = get_embedding_service()

    embedded_chunks = embedding_service.embed_chunks(chunks)

    assert len(embedded_chunks) == len(chunks), "Embedding count mismatch."

    dimensions = {len(item.embedding) for item in embedded_chunks}

    assert len(dimensions) == 1, "Embedding dimensions are inconsistent."

    dimension = dimensions.pop()

    print(f"✅ Embeddings Generated  : {len(embedded_chunks)}")
    print(f"✅ Embedding Dimension   : {dimension}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()

    print("=" * 70)

    print("SUMMARY")

    print("=" * 70)

    print(f"Documents                : {len(documents)}")
    print(f"Chunks                   : {len(chunks)}")
    print(f"Average Chunks/Document  : {len(chunks) / len(documents):.2f}")
    print(f"Embedding Dimension      : {dimension}")

    print()

    print("Sample Chunks")

    print("-" * 70)

    for chunk in chunks[:5]:
        print()

        print(f"{chunk.id}")

        print(chunk.content[:300])

        print("-" * 70)

    print()

    print("🎉 STORY 3 VERIFIED")


if __name__ == "__main__":
    verify_story3()
