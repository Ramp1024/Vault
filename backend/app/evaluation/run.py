"""Run the retrieval evaluation and print a report.

Builds a ``SearchEngine`` configured with enough retrieval depth to score
Recall@10, then evaluates the golden dataset purely through the public
``SearchEngine.search(query)`` API.

Run:  python -m app.evaluation.run [dataset.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.paths import DEFAULT_DATASET_PATH
from app.evaluation.report import format_report
from app.evaluation.runner import RetrievalEvaluator
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.query_analyzer import RuleBasedQueryAnalyzer
from app.search import SearchEngine, VectorSearchStrategy
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

# Deep enough to score Recall@10 and to reveal relevant docs ranked past top-5.
RETRIEVAL_DEPTH = 10


def _build_search_engine() -> SearchEngine:
    qdrant = QdrantService(get_qdrant_client())
    embedding_service = EmbeddingService()

    try:
        fields, multi_fields = qdrant.discover_property_fields()
        registry = (
            MetadataRegistry.from_indexed_fields(fields, multi_fields)
            if fields
            else default_metadata_registry()
        )
    except Exception:
        registry = default_metadata_registry()

    analyzer = RuleBasedQueryAnalyzer(registry=registry, default_top_k=RETRIEVAL_DEPTH)
    return SearchEngine(
        query_analyzer=analyzer,
        strategies=[
            VectorSearchStrategy(
                embedding_service=embedding_service, qdrant_service=qdrant
            )
        ],
    )


def main(dataset_path: Path = DEFAULT_DATASET_PATH) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)
    evaluator = RetrievalEvaluator(search_engine=_build_search_engine())
    report = evaluator.evaluate(dataset)
    print(format_report(report))


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    main(path)
