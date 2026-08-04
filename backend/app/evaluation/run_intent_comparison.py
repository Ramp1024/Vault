"""Benchmark rule-based vs schema-aware composite (rule + LLM) query analysis.

This is the milestone's conversational-intent benchmark. It runs a dataset of
natural-language queries through two otherwise-identical hybrid retrieval
pipelines that differ only in their query analyzer:

    Rule-based            RuleBasedQueryAnalyzer only
    Composite (Rule+LLM)  CompositeQueryAnalyzer(rule-based + LLMIntentAnalyzer)

Everything downstream of query analysis (vector + BM25 retrieval, RRF fusion) is
shared and unchanged, so any difference in Recall@5 / Recall@10 / MRR is
attributable solely to schema-aware intent understanding — exactly the value
this milestone sets out to demonstrate.

The LLM analyzer requires a reachable LLM backend (``LLM_BACKEND``); the schema
is loaded from the persisted schema store, falling back to discovery over the
indexed payload fields.

Run:  python -m app.evaluation.run_intent_comparison [dataset.json] [--markdown]
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.config import settings
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.report import format_overall_comparison, format_report
from app.evaluation.runner import RetrievalEvaluator
from app.processors.llm_intent_analyzer import LLMIntentAnalyzer
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.query_analyzer import (
    CompositeQueryAnalyzer,
    RuleBasedQueryAnalyzer,
)
from app.processors.schema_discovery import schema_from_indexed_fields
from app.search import (
    BM25SearchStrategy,
    RetrievalMode,
    SearchEngine,
    VectorSearchStrategy,
    build_search_engine,
)
from app.services.bm25_index import RankBM25Index
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.llm import build_intent_llm
from app.services.metadata_schema_store import MetadataSchemaStore
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

# Deep enough to score Recall@10 and reveal relevant docs ranked past top-5.
RETRIEVAL_DEPTH = 10

# Default dataset for this benchmark: conversational, natural-language queries.
DEFAULT_DATASET_PATH = (
    Path(__file__).parent / "data" / "conversational_dataset.json"
)


def _discover_fields(qdrant: QdrantService) -> tuple[list[str], set[str]]:
    try:
        return qdrant.discover_property_fields()
    except Exception:
        return [], set()


def _build_registry(fields: list[str], multi_fields: set[str]) -> MetadataRegistry:
    if fields:
        return MetadataRegistry.from_indexed_fields(fields, multi_fields)
    return default_metadata_registry()


def _load_schema(fields: list[str], multi_fields: set[str]):
    schema = MetadataSchemaStore().load()
    if schema:
        return schema
    return schema_from_indexed_fields(fields, multi_fields)


def _build_engine(analyzer, vector_strategy, bm25_strategy) -> SearchEngine:
    return build_search_engine(
        RetrievalMode.HYBRID,
        analyzer,
        vector_strategy=vector_strategy,
        bm25_strategy=bm25_strategy,
        rrf_k=settings.RRF_K,
    )


def main(dataset_path: Path = DEFAULT_DATASET_PATH, *, markdown: bool = False) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)

    qdrant = QdrantService(get_qdrant_client())
    fields, multi_fields = _discover_fields(qdrant)

    # Rebuild BM25 from the exact indexed chunks so both pipelines see the same
    # logical corpus.
    index = RankBM25Index()
    indexed = BM25Indexer(qdrant_service=qdrant, index=index).rebuild()
    print(f"BM25 index built over {indexed} chunks")

    registry = _build_registry(fields, multi_fields)
    schema = _load_schema(fields, multi_fields)
    print(f"Metadata schema: {len(schema.fields)} filterable fields")
    print(f"LLM backend: {settings.LLM_BACKEND} (model {settings.LLM_MODEL})\n")

    # Shared, expensive collaborators — identical for both pipelines.
    vector_strategy = VectorSearchStrategy(
        embedding_service=EmbeddingService(), qdrant_service=qdrant
    )
    bm25_strategy = BM25SearchStrategy(index=index)

    rule_based = RuleBasedQueryAnalyzer(
        registry=registry, default_top_k=RETRIEVAL_DEPTH
    )
    composite = CompositeQueryAnalyzer(
        rule_based=RuleBasedQueryAnalyzer(
            registry=registry, default_top_k=RETRIEVAL_DEPTH
        ),
        llm_based=LLMIntentAnalyzer(
            build_intent_llm(), schema, default_top_k=RETRIEVAL_DEPTH
        ),
    )

    engines = {
        "Rule-based": _build_engine(rule_based, vector_strategy, bm25_strategy),
        "Composite (Rule+LLM)": _build_engine(
            composite, vector_strategy, bm25_strategy
        ),
    }

    reports = {
        name: RetrievalEvaluator(search_engine=engine).evaluate(dataset)
        for name, engine in engines.items()
    }

    print("=" * 90)
    print("CONVERSATIONAL INTENT BENCHMARK  (Rule-based | Composite Rule+LLM)")
    print("=" * 90)
    print(f"Queries: {len(dataset)}\n")
    print("Overall metrics and composite deltas:")
    print(
        format_overall_comparison(
            reports, target="Composite (Rule+LLM)", markdown=markdown
        )
    )
    print()
    for name, report in reports.items():
        print(f"--- {name} ---")
        print(format_report(report))
        print()
    print("=" * 90)


if __name__ == "__main__":
    args = sys.argv[1:]
    want_markdown = "--markdown" in args
    positional = [arg for arg in args if not arg.startswith("--")]
    path = Path(positional[0]) if positional else DEFAULT_DATASET_PATH
    main(path, markdown=want_markdown)
