import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.models.search_result import SearchResult


class QdrantService:
    def __init__(self, client: QdrantClient):
        self.client = client

    def _chunk_filter(self, chunk_id: str) -> Filter:
        return Filter(
            must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
        )

    def _point_id_for_chunk(self, chunk_id: str) -> str:
        """Create a deterministic UUID point id from an arbitrary chunk id string."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _chunk_from_payload(self, payload: dict[str, Any]) -> Chunk:
        """Reconstruct a domain Chunk from a stored Qdrant payload."""
        required_keys = {
            "chunk_id",
            "document_id",
            "document_title",
            "chunk_index",
            "content",
        }
        missing_keys = required_keys - payload.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise RuntimeError(
                f"Stored chunk payload missing required fields: {missing}"
            )

        metadata = {
            key: value for key, value in payload.items() if key not in required_keys
        }

        return Chunk(
            id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            document_title=str(payload["document_title"]),
            content=str(payload["content"]),
            chunk_index=int(payload["chunk_index"]),
            metadata=metadata,
        )

    def _ensure_collection(self, vector_size: int) -> None:
        """Create collection if it does not exist."""
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def collection_exists(self) -> bool:
        """Return True if the configured Qdrant collection exists."""
        return self.client.collection_exists(
            collection_name=settings.QDRANT_COLLECTION_NAME
        )

    def count(self) -> int:
        """Return point count for the configured Qdrant collection."""
        result = self.client.count(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            exact=True,
        )
        return int(result.count)

    def get_point(self, point_id: str):
        """Fetch a single point by original chunk id from the configured collection."""
        points, _ = self.client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=self._chunk_filter(point_id),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        return points[0] if points else None

    def chunk_exists(self, chunk_id: str) -> bool:
        """Return True if a point exists for the given original chunk id."""
        points, _ = self.client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=self._chunk_filter(chunk_id),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    def health_check(self) -> bool:
        """
        Check if the Qdrant service is reachable.
        Returns True if the service is reachable, False otherwise.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Backward-compatible upsert entrypoint.

        Delegates to batched upsert logic.

        Args:
            embedded_chunks: List of EmbeddedChunk objects to upsert

        Raises:
            RuntimeError: If the upsert operation fails
        """
        self.upsert_batch(embedded_chunks)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search for semantically similar chunks by query embedding."""
        if not query_embedding:
            return []
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        try:
            response = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [
                SearchResult(
                    chunk=self._chunk_from_payload(point.payload or {}),
                    score=float(point.score),
                )
                for point in response.points
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to search Qdrant: {e}")

    def upsert_batch(
        self,
        embedded_chunks: list[EmbeddedChunk],
        batch_size: int | None = None,
    ) -> int:
        """Upsert embedded chunks into Qdrant in batches.

        Args:
            embedded_chunks: Embedded chunks to write
            batch_size: Optional override for batch size. If omitted,
                `settings.QDRANT_UPSERT_BATCH_SIZE` is used.

        Returns:
            Number of points written.

        Raises:
            RuntimeError: If the upsert operation fails
            ValueError: If batch_size is not positive
        """
        if not embedded_chunks:
            return 0

        effective_batch_size = batch_size or settings.QDRANT_UPSERT_BATCH_SIZE
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        vector_size = len(embedded_chunks[0].embedding)
        if vector_size <= 0:
            raise ValueError("embedding vectors must not be empty")

        total = len(embedded_chunks)

        try:
            self._ensure_collection(vector_size=vector_size)

            for i in range(0, total, effective_batch_size):
                batch = embedded_chunks[i : i + effective_batch_size]
                points = [
                    {
                        "id": self._point_id_for_chunk(item.chunk.id),
                        "vector": item.embedding,
                        "payload": {
                            "chunk_id": item.chunk.id,
                            "document_id": item.chunk.document_id,
                            "document_title": item.chunk.document_title,
                            "chunk_index": item.chunk.chunk_index,
                            "content": item.chunk.content,
                            **item.chunk.metadata,
                        },
                    }
                    for item in batch
                ]

                self.client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=points,
                )

            return total
        except Exception as e:
            raise RuntimeError(f"Failed to upsert chunks into Qdrant: {e}")
