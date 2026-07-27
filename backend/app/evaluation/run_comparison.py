"""Benchmark Vector, BM25, and Hybrid (RRF) retrieval on the golden dataset.

Runs the same golden dataset through three ``SearchEngine`` configurations built
by :func:`app.search.build_search_engine` — Vector only, BM25 only, and Hybrid
(Vector + BM25 fused with Reciprocal Rank Fusion) — purely through the public
``SearchEngine.search(query)`` API. The BM25 index is (re)built from the chunks
already stored in Qdrant so every mode sees the same logical chunks.

Reports overall metrics per mode, a per-category breakdown, and Hybrid-vs-single
deltas. Pass ``--diagnostics`` to also print a per-query, per-mode markdown
report for manual inspection.

Run:  python -m app.evaluation.run_comparison [dataset.json] [--diagnostics]
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.paths import DEFAULT_DATASET_PATH
from app.evaluation.report import (
    format_category_comparison,
    format_overall_comparison,
    format_query_diagnostics,
)
from app.evaluation.runner import RetrievalEvaluator
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.query_analyzer import RuleBasedQueryAnalyzer
from app.search import RetrievalMode, build_search_engine
from app.search.strategy import BM25SearchStrategy, VectorSearchStrategy
from app.services.bm25_index import RankBM25Index
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

# Deep enough to score Recall@10 and reveal relevant docs ranked past top-5.
RETRIEVAL_DEPTH = 10


def _build_analyzer(qdrant: QdrantService) -> RuleBasedQueryAnalyzer:
    try:
        fields, multi_fields = qdrant.discover_property_fields()
        registry = (
            MetadataRegistry.from_indexed_fields(fields, multi_fields)
            if fields
            else default_metadata_registry()
        )
    except Exception:
        registry = default_metadata_registry()
    return RuleBasedQueryAnalyzer(registry=registry, default_top_k=RETRIEVAL_DEPTH)


def main(
    dataset_path: Path = DEFAULT_DATASET_PATH, *, diagnostics: bool = False
) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)

    qdrant = QdrantService(get_qdrant_client())
    analyzer = _build_analyzer(qdrant)

    # Rebuild the BM25 index from the exact chunks stored in Qdrant so every mode
    # is evaluated over the same logical chunks.
    index = RankBM25Index()
    indexed = BM25Indexer(qdrant_service=qdrant, index=index).rebuild()
    print(f"BM25 index built over {indexed} chunks\n")

    # Share the expensive collaborators across all modes.
    vector_strategy = VectorSearchStrategy(
        embedding_service=EmbeddingService(), qdrant_service=qdrant
    )
    bm25_strategy = BM25SearchStrategy(index=index)

    engines = {
        "Vector": build_search_engine(
            RetrievalMode.VECTOR, analyzer, vector_strategy=vector_strategy
        ),
        "BM25": build_search_engine(
            RetrievalMode.BM25, analyzer, bm25_strategy=bm25_strategy
        ),
        "Hybrid": build_search_engine(
            RetrievalMode.HYBRID,
            analyzer,
            vector_strategy=vector_strategy,
            bm25_strategy=bm25_strategy,
        ),
    }

    reports = {
        name: RetrievalEvaluator(search_engine=engine).evaluate(dataset)
        for name, engine in engines.items()
    }

    print("=" * 90)
    print("RETRIEVAL MODE COMPARISON  (Vector | BM25 | Hybrid RRF)")
    print("=" * 90)
    print(f"Queries: {len(dataset)}\n")

    print("Overall metrics and Hybrid deltas:")
    print(format_overall_comparison(reports, target="Hybrid"))
    print()
    print("Per-category breakdown:")
    print(format_category_comparison(reports))
    print("=" * 90)

    if diagnostics:
        print()
        print(format_query_diagnostics(dataset, engines))


if __name__ == "__main__":
    args = sys.argv[1:]
    want_diagnostics = "--diagnostics" in args
    positional = [arg for arg in args if not arg.startswith("--")]
    path = Path(positional[0]) if positional else DEFAULT_DATASET_PATH
    main(path, diagnostics=want_diagnostics)
