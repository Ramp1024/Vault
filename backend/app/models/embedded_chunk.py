from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]
