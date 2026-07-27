from __future__ import annotations

from app.services.bm25_index import BM25Index, RankBM25Index
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService


class BM25Indexer:
    """Builds and persists the BM25 index from the chunks stored in Qdrant.

    This is the integration point between the existing indexing pipeline and the
    lexical index. Rather than introducing a separate document model, it sources
    the exact chunks already indexed in the vector store, so the BM25 index and
    the vector index always reference the same logical chunks (same ids,
    document ids, and metadata). Rebuild after a sync to keep them in step.
    """

    def __init__(
        self,
        qdrant_service: QdrantService | None = None,
        index: BM25Index | None = None,
    ) -> None:
        self.qdrant_service = qdrant_service or QdrantService(get_qdrant_client())
        self.index = index or RankBM25Index()

    def rebuild(self) -> int:
        """Rebuild the BM25 index from all Qdrant chunks and persist it.

        Returns the number of chunks indexed.
        """
        chunks = self.qdrant_service.fetch_all_chunks()
        self.index.build(chunks)
        self.index.save()
        return self.index.size
