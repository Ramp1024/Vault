"""Benchmark Vector, BM25, Hybrid (RRF), and Hybrid + Cross-Encoder retrieval.

Runs the golden dataset through four retrieval pipelines built from the same
shared collaborators:

    Vector | BM25 | Hybrid (RRF) | Hybrid + Cross-Encoder Reranker

Each pipeline is executed once through an :class:`InstrumentedPipeline`, which
mirrors ``SearchEngine.search`` while timing every stage (query analysis, vector
retrieval, BM25 retrieval, fusion, cross-encoder), so the report covers both
quality (Recall@5/@10, MRR) and latency without changing the frozen
``SearchEngine`` API.

Reports overall metrics, a per-category breakdown, a headline quality-vs-latency
table, and a per-stage latency breakdown. Pass ``--diagnostics`` to also print a
per-query reranking movement report (how the cross-encoder reordered candidates).
Pass ``--markdown`` to render the tables as GitHub-flavored markdown.

Run:  python -m app.evaluation.run_comparison [dataset.json] [--diagnostics] [--markdown]
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.config import settings
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.paths import DEFAULT_DATASET_PATH
from app.evaluation.pipeline import InstrumentedPipeline, run_pipeline
from app.evaluation.report import (
    format_category_comparison,
    format_latency_breakdown,
    format_overall_comparison,
    format_performance_table,
    format_query_diagnostics,
    format_rerank_diagnostics,
)
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.query_analyzer import RuleBasedQueryAnalyzer
from app.search.fusion import IdentityFusionStrategy, ReciprocalRankFusion
from app.search.reranker import CrossEncoderReranker
from app.search.strategy import BM25SearchStrategy, VectorSearchStrategy
from app.services.bm25_index import RankBM25Index
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

# Retrieve a candidate pool deep enough for the cross-encoder to rerank; the same
# depth is used by every pipeline so the comparison is fair (the reranker only
# reorders the pool the hybrid stage already produced).
RETRIEVAL_DEPTH = settings.RERANK_CANDIDATE_POOL


class _ResultsView:
    """Adapts a pipeline run's captured results to the ``search(query)`` shape.

    Lets the multi-pipeline diagnostics reuse already-computed results instead of
    re-running retrieval (which would re-invoke the cross-encoder).
    """

    def __init__(self, dataset: EvaluationDataset, run) -> None:
        self._by_query = {
            case.query: run.results_by_case.get(case.id, []) for case in dataset.cases
        }

    def search(self, query: str):
        return self._by_query.get(query, [])


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
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    diagnostics: bool = False,
    markdown: bool = False,
) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)

    qdrant = QdrantService(get_qdrant_client())
    analyzer = _build_analyzer(qdrant)

    # Rebuild the BM25 index from the exact chunks stored in Qdrant so every
    # pipeline is evaluated over the same logical chunks.
    index = RankBM25Index()
    indexed = BM25Indexer(qdrant_service=qdrant, index=index).rebuild()
    print(f"BM25 index built over {indexed} chunks")

    # Share the expensive collaborators across all pipelines.
    vector_strategy = VectorSearchStrategy(
        embedding_service=EmbeddingService(), qdrant_service=qdrant
    )
    bm25_strategy = BM25SearchStrategy(index=index)

    # For the benchmark the reranker returns the full reordered pool (top_n=None)
    # so Recall@10 remains measurable on the reranked results.
    reranker = CrossEncoderReranker(
        settings.RERANK_MODEL, candidate_pool=RETRIEVAL_DEPTH, top_n=None
    )

    pipelines = {
        "Vector": InstrumentedPipeline(
            analyzer, [("Vector", vector_strategy)], IdentityFusionStrategy()
        ),
        "BM25": InstrumentedPipeline(
            analyzer, [("BM25", bm25_strategy)], IdentityFusionStrategy()
        ),
        "Hybrid": InstrumentedPipeline(
            analyzer,
            [("Vector", vector_strategy), ("BM25", bm25_strategy)],
            ReciprocalRankFusion(k=settings.RRF_K),
        ),
        "Hybrid + Reranker": InstrumentedPipeline(
            analyzer,
            [("Vector", vector_strategy), ("BM25", bm25_strategy)],
            ReciprocalRankFusion(k=settings.RRF_K),
            reranker=reranker,
        ),
    }

    print(f"Reranker model: {settings.RERANK_MODEL} (loading on first rerank)\n")

    runs = {
        name: run_pipeline(name, pipeline, dataset)
        for name, pipeline in pipelines.items()
    }
    reports = {name: run.report for name, run in runs.items()}
    ordered_runs = list(runs.values())

    print("=" * 90)
    print("RETRIEVAL PIPELINE COMPARISON  (Vector | BM25 | Hybrid | Hybrid + Reranker)")
    print("=" * 90)
    print(f"Queries: {len(dataset)}\n")

    print("Quality vs latency:")
    print(format_performance_table(ordered_runs, markdown=markdown))
    print()
    print("Per-stage latency (ms):")
    print(format_latency_breakdown(ordered_runs, markdown=markdown))
    print()
    print("Overall metrics and reranker deltas:")
    print(
        format_overall_comparison(
            reports, target="Hybrid + Reranker", markdown=markdown
        )
    )
    print()
    print("Per-category breakdown:")
    print(format_category_comparison(reports, markdown=markdown))
    print("=" * 90)

    if diagnostics:
        print()
        print(
            format_rerank_diagnostics(
                dataset, runs["Hybrid"], runs["Hybrid + Reranker"]
            )
        )
        print()
        engines_view = {name: _ResultsView(dataset, run) for name, run in runs.items()}
        print(format_query_diagnostics(dataset, engines_view))


if __name__ == "__main__":
    args = sys.argv[1:]
    want_diagnostics = "--diagnostics" in args
    want_markdown = "--markdown" in args
    positional = [arg for arg in args if not arg.startswith("--")]
    path = Path(positional[0]) if positional else DEFAULT_DATASET_PATH
    main(path, diagnostics=want_diagnostics, markdown=want_markdown)
