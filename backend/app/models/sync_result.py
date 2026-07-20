from dataclasses import dataclass


@dataclass(frozen=True)
class SyncResult:
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    vectors_upserted: int
    duration: float
    documents_skipped: int = 0