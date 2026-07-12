"""Retrieval evaluation using a JSON benchmark dataset.

Dataset format:
[
  {
    "query": "webpack",
    "expected_titles": ["Webpack"],
    "top_k": 3
  }
]
"""

from dataclasses import dataclass
import json
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


DATASET_PATH = Path(__file__).with_name("retrieval_dataset.json")
MIN_HIT_AT_K = 0.80
MIN_MRR = 0.70


@dataclass(frozen=True)
class RetrievalSample:
    query: str
    expected_titles: list[str]
    top_k: int


@dataclass(frozen=True)
class QueryEvalResult:
    query: str
    expected_titles: list[str]
    retrieved_titles: list[str]
    hit: bool
    reciprocal_rank: float


def _normalize_title(value: str) -> str:
    return value.strip().lower()


def _load_dataset(path: Path) -> list[RetrievalSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array")

    samples: list[RetrievalSample] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each dataset item must be a JSON object")

        query = item.get("query")
        expected_titles = item.get("expected_titles")
        top_k = item.get("top_k", 3)

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each dataset item needs a non-empty string 'query'")
        if (
            not isinstance(expected_titles, list)
            or not expected_titles
            or not all(isinstance(title, str) and title.strip() for title in expected_titles)
        ):
            raise ValueError(
                "Each dataset item needs a non-empty string array 'expected_titles'"
            )
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("Each dataset item needs a positive integer 'top_k'")

        samples.append(
            RetrievalSample(
                query=query,
                expected_titles=expected_titles,
                top_k=top_k,
            )
        )

    if not samples:
        raise ValueError("Dataset cannot be empty")

    return samples


def _ensure_index(qdrant: QdrantService, embedding_service: EmbeddingService) -> None:
    if qdrant.collection_exists() and qdrant.count() > 0:
        return

    print("Indexing chunks because collection is empty...")
    documents = NotionConnector().ingest()
    chunks = Chunker().chunk_documents(documents)
    embedded_chunks = embedding_service.embed_chunks(chunks)
    qdrant.upsert(embedded_chunks)


def _evaluate_samples(
    samples: list[RetrievalSample],
    qdrant: QdrantService,
    embedding_service: EmbeddingService,
) -> list[QueryEvalResult]:
    results: list[QueryEvalResult] = []

    for sample in samples:
        query_embedding = embedding_service.embed(sample.query)
        search_results = qdrant.search(query_embedding, limit=sample.top_k)

        retrieved_titles = [result.chunk.document_title for result in search_results]
        normalized_expected = {_normalize_title(title) for title in sample.expected_titles}

        matched_rank: int | None = None
        for index, title in enumerate(retrieved_titles, start=1):
            if _normalize_title(title) in normalized_expected:
                matched_rank = index
                break

        hit = matched_rank is not None
        reciprocal_rank = 0.0 if matched_rank is None else 1.0 / matched_rank

        results.append(
            QueryEvalResult(
                query=sample.query,
                expected_titles=sample.expected_titles,
                retrieved_titles=retrieved_titles,
                hit=hit,
                reciprocal_rank=reciprocal_rank,
            )
        )

    return results


def verify_retrieval_evaluation(dataset_path: Path = DATASET_PATH) -> None:
    print("=" * 90)
    print("VAULT - RETRIEVAL EVALUATION")
    print("=" * 90)

    samples = _load_dataset(dataset_path)

    embedding_service = EmbeddingService()
    qdrant = QdrantService(client=get_qdrant_client())

    _ensure_index(qdrant, embedding_service)
    eval_results = _evaluate_samples(samples, qdrant, embedding_service)

    total = len(eval_results)
    hit_count = sum(1 for result in eval_results if result.hit)
    hit_at_k = hit_count / total
    mrr = sum(result.reciprocal_rank for result in eval_results) / total

    print(f"Dataset: {dataset_path}")
    print(f"Queries: {total}")
    print("-" * 90)

    failures: list[QueryEvalResult] = []

    for result in eval_results:
        status = "PASS" if result.hit else "FAIL"
        print(f"[{status}] query='{result.query}'")
        print(f"       expected_titles={result.expected_titles}")
        print(f"       retrieved_titles={result.retrieved_titles}")
        if not result.hit:
            failures.append(result)

    print("-" * 90)
    print(f"hit@k: {hit_at_k:.4f} ({hit_count}/{total})")
    print(f"MRR:   {mrr:.4f}")

    assert hit_at_k >= MIN_HIT_AT_K, (
        f"hit@k {hit_at_k:.4f} is below required threshold {MIN_HIT_AT_K:.2f}"
    )
    assert mrr >= MIN_MRR, f"MRR {mrr:.4f} is below required threshold {MIN_MRR:.2f}"

    if failures:
        print(f"\nNote: {len(failures)} query/queries did not hit expected titles in top_k")

    print("✅ Retrieval evaluation passed")


if __name__ == "__main__":
    verify_retrieval_evaluation()
