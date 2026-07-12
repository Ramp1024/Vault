"""Story 4 - Milestone 3 Verification.

Verifies semantic retrieval over indexed chunks stored in Qdrant.
"""

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.connectors.notion.connector import NotionConnector
from app.processors.chunker import Chunker
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService


QUERIES = [
    "docker networking",
    "docker compose",
    "webpack",
    "tree sitter",
    "apache arrow",
    "shoulder workout",
    "lat pulldown",
]


def _ensure_index(qdrant: QdrantService, embedding_service: EmbeddingService) -> None:
    if qdrant.collection_exists() and qdrant.count() > 0:
        return

    print("Indexing chunks for search verification...")
    documents = NotionConnector().ingest()
    chunks = Chunker().chunk_documents(documents)
    embedded_chunks = embedding_service.embed_chunks(chunks)
    qdrant.upsert(embedded_chunks)


def verify_search() -> None:
    print("=" * 80)
    print("VAULT - SEMANTIC SEARCH VERIFICATION")
    print("=" * 80)

    embedding_service = EmbeddingService()
    qdrant = QdrantService(client=get_qdrant_client())

    _ensure_index(qdrant, embedding_service)

    for query in QUERIES:
        print(f"\nQuery: {query}")
        query_embedding = embedding_service.embed(query)
        results = qdrant.search(query_embedding, limit=5)

        assert results, f"No search results returned for query: {query}"

        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending for query: {query}"
        )

        for index, result in enumerate(results, start=1):
            assert result.chunk.content.strip(), (
                f"Empty chunk content returned for query: {query}"
            )
            assert qdrant.chunk_exists(result.chunk.id), (
                f"Chunk id missing from index: {result.chunk.id}"
            )

            preview = result.chunk.content.replace("\n", " ")[:250]
            print(f"  [{index}] score={result.score:.4f}")
            print(f"      title={result.chunk.document_title}")
            print(f"      chunk_index={result.chunk.chunk_index}")
            print(f"      preview={preview}")

    print("\n✅ Semantic retrieval verified")


if __name__ == "__main__":
    verify_search()