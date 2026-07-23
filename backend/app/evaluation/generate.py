"""Generate an initial golden retrieval dataset from the indexed Vault corpus.

This is a curation aid, not part of the evaluation framework itself: it inspects
the index (via ``QdrantService``) to build ground-truth expectations, then writes
a human-readable JSON dataset that can be manually reviewed and edited.

Run:  python -m app.evaluation.generate
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.paths import DEFAULT_DATASET_PATH
from app.processors.metadata_registry import _decamel
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

# Categorical fields worth turning into metadata/hybrid cases. Free-text and
# per-day fields (notes, personalWin, leetcode, ...) are excluded because their
# values are not good filter targets.
CATEGORICAL_FIELDS = ("category", "leetcodeTopic", "status", "project", "team", "tags")

# Weekday-style titles repeat across daily notes and are not distinctive.
_WEEKDAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


@dataclass
class DocInfo:
    document_id: str
    title: str
    chunk_ids: list[str]
    properties: dict


def _load_corpus(qdrant: QdrantService) -> dict[str, DocInfo]:
    client = qdrant.client
    from app.core.config import settings

    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    docs: dict[str, DocInfo] = {}
    chunks_by_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("document_id")
        if not doc_id:
            continue
        chunk_id = payload.get("chunk_id")
        index = int(payload.get("chunk_index", 0))
        chunks_by_doc[doc_id].append((index, str(chunk_id)))
        if doc_id not in docs:
            docs[doc_id] = DocInfo(
                document_id=doc_id,
                title=str(payload.get("document_title", "")),
                chunk_ids=[],
                properties=payload.get("properties") or {},
            )

    for doc_id, info in docs.items():
        info.chunk_ids = [cid for _, cid in sorted(chunks_by_doc[doc_id])]
    return docs


def _value_to_docs(docs: dict[str, DocInfo]) -> dict[tuple[str, str], list[str]]:
    """Map (field, value) -> sorted document ids that carry it."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for info in docs.values():
        for field, value in info.properties.items():
            if field not in CATEGORICAL_FIELDS:
                continue
            values = value if isinstance(value, list) else [value]
            for single in values:
                if isinstance(single, str) and single.strip():
                    index[(field, single)].add(info.document_id)
    return {key: sorted(ids) for key, ids in index.items()}


def _chunks_for(docs: dict[str, DocInfo], doc_ids: list[str]) -> list[str]:
    chunk_ids: list[str] = []
    for doc_id in doc_ids:
        chunk_ids.extend(docs[doc_id].chunk_ids)
    return chunk_ids


def _slug(text: str) -> str:
    return "-".join("".join(c if c.isalnum() else " " for c in text).split()).lower()


def _unique_title_docs(docs: dict[str, DocInfo]) -> list[DocInfo]:
    counts: dict[str, int] = defaultdict(int)
    for info in docs.values():
        counts[info.title.strip().lower()] += 1
    unique = [
        info
        for info in docs.values()
        if counts[info.title.strip().lower()] == 1
        and len(info.title.strip()) >= 6
        and info.title.strip().lower() not in _WEEKDAYS
    ]
    # Prefer richer documents (more chunks) as clearer semantic targets.
    return sorted(unique, key=lambda d: (-len(d.chunk_ids), d.title.lower()))


def generate_dataset(docs: dict[str, DocInfo]) -> EvaluationDataset:
    cases: list[EvaluationCase] = []
    value_index = _value_to_docs(docs)

    # 1. Metadata cases: exact filter ground truth.
    metadata_pairs = sorted(value_index.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    metadata_pairs = [(k, v) for k, v in metadata_pairs if 1 <= len(v) <= 8]
    for (field, value), doc_ids in metadata_pairs[:8]:
        label = _decamel(field)
        cases.append(
            EvaluationCase(
                id=f"metadata-{_slug(field)}-{_slug(value)}",
                query=f"{label}: {value}",
                category="metadata",
                expected_documents=tuple(doc_ids),
                expected_chunks=tuple(_chunks_for(docs, doc_ids)),
            )
        )

    unique_docs = _unique_title_docs(docs)

    # 2. Keyword cases: distinctive title as a keyword query.
    for info in unique_docs[:6]:
        cases.append(
            EvaluationCase(
                id=f"keyword-{_slug(info.title)}"[:60],
                query=info.title.strip(),
                category="keyword",
                expected_documents=(info.document_id,),
                expected_chunks=tuple(info.chunk_ids),
            )
        )

    # 3. Semantic / natural-language questions about distinctive documents.
    for info in unique_docs[:6]:
        cases.append(
            EvaluationCase(
                id=f"semantic-{_slug(info.title)}"[:60],
                query=f"What do my notes say about {info.title.strip()}?",
                category="semantic",
                expected_documents=(info.document_id,),
                expected_chunks=tuple(info.chunk_ids),
            )
        )

    # 4. Hybrid cases: semantic phrase + metadata filter (filter is ground truth).
    for (field, value), doc_ids in metadata_pairs[:4]:
        label = _decamel(field)
        cases.append(
            EvaluationCase(
                id=f"hybrid-{_slug(field)}-{_slug(value)}",
                query=f"what did I work on {label}: {value}",
                category="hybrid",
                expected_documents=tuple(doc_ids),
                expected_chunks=tuple(_chunks_for(docs, doc_ids)),
            )
        )

    return EvaluationDataset(cases=tuple(cases))


def main(output_path: Path = DEFAULT_DATASET_PATH) -> None:
    qdrant = QdrantService(get_qdrant_client())
    if not qdrant.collection_exists() or qdrant.count() == 0:
        raise SystemExit("Qdrant collection is empty; index the corpus before generating.")

    docs = _load_corpus(qdrant)
    dataset = generate_dataset(docs)
    dataset.to_file(output_path)
    print(f"Wrote {len(dataset)} cases to {output_path}")
    by_category: dict[str, int] = defaultdict(int)
    for case in dataset.cases:
        by_category[case.category or "uncategorized"] += 1
    for category, count in sorted(by_category.items()):
        print(f"  {category:<10} {count}")


if __name__ == "__main__":
    main()
