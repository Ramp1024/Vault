"""Embedding service for text and document chunks using Ollama."""

import httpx

from app.core.config import settings
from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk


class EmbeddingService:
    """Generate embeddings using Ollama and a configured embedding model.

    This service provides a simple, focused interface for embedding:
    - Individual text strings
    - EmbeddedChunk objects for vector-store ingestion

    The embedding model is configured via the settings layer.
    """

    MODEL = settings.OLLAMA_EMBED_MODEL

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=settings.OLLAMA_TIMEOUT_SECONDS,
        )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
                text: Text to embed

        Returns:
                List of floats representing the embedding vector (768 dimensions)

        Raises:
                RuntimeError: If the request to Ollama fails or the response is invalid
        """
        try:
            response = self.client.post(
                "/api/embed",
                json={
                    "model": self.MODEL,
                    "input": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])

            if embeddings:
                return embeddings[0]

            raise RuntimeError("Failed to embed text: Ollama returned no embeddings")
        except (httpx.HTTPError, ValueError) as e:
            raise RuntimeError(f"Failed to embed text: {e}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed multiple chunks efficiently.

        Args:
                chunks: List of Chunk objects to embed

        Returns:
                List of EmbeddedChunk objects (one per chunk)

        Raises:
                RuntimeError: If one or more chunk embedding requests fail
        """
        if not chunks:
            return []

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk in chunks:
            embedding = self.embed(chunk.content)
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

        return embedded_chunks


def get_embedding_service() -> EmbeddingService:
    """Factory function to create an embedding service instance."""
    return EmbeddingService()
