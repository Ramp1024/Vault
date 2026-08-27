"""Run the production retrieval benchmark across every query analyzer.

The framework's core question is "does query understanding help, and by how
much?". To answer it, this runner evaluates the same large, realistic dataset
through three otherwise-identical hybrid pipelines that differ ONLY in their
query analyzer:

    Rule-based            deterministic ``field: value`` parsing, no LLM
    LLM-only              schema-aware LLM intent understanding on its own
    Composite (Rule+LLM)  rule-based precision merged with LLM understanding

Everything downstream of analysis (vector + BM25 retrieval, RRF fusion, optional
reranking) is shared and unchanged, so every reported difference in Recall@5 /
Recall@10 / MRR, filter recall, intent accuracy, or latency is attributable
solely to the analyzer — which is exactly the improvement the report makes
visible, side by side, for every category and difficulty.

It also reports why hybrid retrieval exists (strategy contribution), per-stage
latency, and a failure analysis that lists the queries each analyzer upgrade
fixes and regresses.

Both LLM analyzers require a reachable LLM backend (``LLM_BACKEND``); the schema
is loaded from the persisted store, falling back to discovery over the indexed
fields. Pass ``--rule-only`` to skip the LLM analyzers, ``--rerank`` to add a
cross-encoder pass, and ``--markdown`` for markdown tables.

Run:  python -m app.evaluation.run_production [dataset.json] [--rule-only] [--rerank] [--markdown]
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.config import settings
from app.evaluation import production_report as pr
from app.evaluation.analysis_metrics import (
    AnalysisReport,
    AnalyzedCase,
    analyze_dataset,
    build_analysis_report,
)
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.hybrid_metrics import hybrid_contribution
from app.evaluation.latency_metrics import StageLatency, run_latencies
from app.evaluation.metrics import default_metrics
from app.evaluation.pipeline import InstrumentedPipeline, PipelineRun, run_pipeline
from app.evaluation.report import format_overall_comparison
from app.models.search_request import SearchRequest
from app.processors.llm_intent_analyzer import LLMIntentAnalyzer
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.query_analyzer import (
    QueryAnalyzer,
    RuleBasedQueryAnalyzer,
)
from app.processors.query_intent import DeterministicIntentAnalyzer
from app.processors.augmenting_analyzer import AugmentingIntentAnalyzer
from app.processors.constraint_proposal import LLMConstraintProposer
from app.processors.constraint_validation import ConstraintValidator
from app.processors.schema_discovery import (
    enrich_schema_with_values,
    schema_from_indexed_fields,
)
from app.search import (
    CrossEncoderReranker,
    IdentityFusionStrategy,
    ReciprocalRankFusion,
    VectorSearchStrategy,
)
from app.search.strategy import BM25SearchStrategy
from app.services.bm25_index import RankBM25Index
from app.services.bm25_indexer import BM25Indexer
from app.services.embedding_service import EmbeddingService
from app.services.llm import build_intent_llm
from app.services.metadata_schema_store import MetadataSchemaStore
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "production_dataset.json"

# Deep enough to score Recall@10 and reveal relevant docs ranked past top-5.
RETRIEVAL_DEPTH = 10
PRIMARY_K = 5

# Analyzer used for the retrieval-strategy (hybrid contribution) breakdown and
# the detailed failure analysis — the production default. The augmenting
# analyzer keeps the deterministic result authoritative and only adds validated,
# grounded LLM constraints, so it can never score below the deterministic run.
REFERENCE_ANALYZER = "Augmenting (Det+LLM)"


class _SafeCachingAnalyzer(QueryAnalyzer):
    """Memoize an analyzer's output per query and degrade gracefully on failure.

    Caching guarantees each (expensive, LLM-backed) analysis runs exactly once
    per unique query even though the query is analyzed by both the retrieval
    pipeline and the query-analysis metrics pass. If the wrapped analyzer raises
    (e.g. the LLM backend is briefly unreachable) it falls back to a plain
    semantic request so a single failure never aborts the whole benchmark.
    """

    def __init__(self, inner: QueryAnalyzer, *, default_top_k: int) -> None:
        self._inner = inner
        self._default_top_k = default_top_k
        self._cache: dict[str, SearchRequest] = {}

    def analyze(self, query: str) -> SearchRequest:
        if query not in self._cache:
            try:
                self._cache[query] = self._inner.analyze(query)
            except Exception:
                self._cache[query] = SearchRequest(
                    semantic_query=query, filters=[], top_k=self._default_top_k
                )
        return self._cache[query]


def _discover(qdrant: QdrantService) -> tuple[list[str], set[str]]:
    try:
        return qdrant.discover_property_fields()
    except Exception:
        return [], set()


def _registry(fields: list[str], multi: set[str]) -> MetadataRegistry:
    if fields:
        return MetadataRegistry.from_indexed_fields(fields, multi)
    return default_metadata_registry()


def _load_schema(fields: list[str], multi: set[str], values: dict[str, list[str]]):
    schema = MetadataSchemaStore().load() or schema_from_indexed_fields(fields, multi)
    return enrich_schema_with_values(schema, values) if values else schema


def _build_analyzers(
    registry: MetadataRegistry,
    fields: list[str],
    multi: set[str],
    values: dict[str, list[str]],
    *,
    rule_only: bool,
) -> dict[str, QueryAnalyzer]:
    """Build every analyzer variant, each wrapped for caching + resilience."""

    def rule() -> RuleBasedQueryAnalyzer:
        return RuleBasedQueryAnalyzer(registry=registry, default_top_k=RETRIEVAL_DEPTH)

    schema = _load_schema(fields, multi, values)

    analyzers: dict[str, QueryAnalyzer] = {
        "Rule-based": _SafeCachingAnalyzer(rule(), default_top_k=RETRIEVAL_DEPTH),
        # Deterministic, schema-aware segmenter (no LLM): lexical route + value
        # extraction + confidence gate.
        "Deterministic": _SafeCachingAnalyzer(
            DeterministicIntentAnalyzer(schema, default_top_k=RETRIEVAL_DEPTH),
            default_top_k=RETRIEVAL_DEPTH,
        ),
    }
    if rule_only:
        return analyzers

    def llm() -> LLMIntentAnalyzer:
        return LLMIntentAnalyzer(
            build_intent_llm(), schema, default_top_k=RETRIEVAL_DEPTH
        )

    def augmenting() -> AugmentingIntentAnalyzer:
        return AugmentingIntentAnalyzer(
            deterministic=DeterministicIntentAnalyzer(
                schema, default_top_k=RETRIEVAL_DEPTH
            ),
            proposer=LLMConstraintProposer(build_intent_llm(), schema),
            validator=ConstraintValidator(
                schema,
                min_confidence=settings.INTENT_LLM_MIN_CONFIDENCE,
                min_grounding=settings.INTENT_GROUNDING_THRESHOLD,
            ),
        )

    analyzers["LLM-only"] = _SafeCachingAnalyzer(
        llm(), default_top_k=RETRIEVAL_DEPTH
    )
    analyzers[REFERENCE_ANALYZER] = _SafeCachingAnalyzer(
        augmenting(),
        default_top_k=RETRIEVAL_DEPTH,
    )
    return analyzers


def _hybrid_pipeline(
    analyzer: QueryAnalyzer,
    vector: VectorSearchStrategy,
    bm25: BM25SearchStrategy,
    *,
    reranker: CrossEncoderReranker | None = None,
) -> InstrumentedPipeline:
    return InstrumentedPipeline(
        analyzer,
        [("Vector", vector), ("BM25", bm25)],
        ReciprocalRankFusion(k=settings.RRF_K),
        reranker=reranker,
    )


def _single_pipeline(
    analyzer: QueryAnalyzer, name: str, strategy
) -> InstrumentedPipeline:
    return InstrumentedPipeline(analyzer, [(name, strategy)], IdentityFusionStrategy())


def main(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    rule_only: bool = False,
    with_rerank: bool = False,
    markdown: bool = False,
) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)
    metrics = default_metrics()

    qdrant = QdrantService(get_qdrant_client())
    fields, multi = _discover(qdrant)
    values = qdrant.discover_property_values()

    index = RankBM25Index()
    indexed = BM25Indexer(qdrant_service=qdrant, index=index).rebuild()

    registry = _registry(fields, multi)
    analyzers = _build_analyzers(registry, fields, multi, values, rule_only=rule_only)

    vector = VectorSearchStrategy(
        embedding_service=EmbeddingService(), qdrant_service=qdrant
    )
    bm25 = BM25SearchStrategy(index=index)

    # One hybrid run per analyzer — the comparison backbone.
    hybrid_runs: dict[str, PipelineRun] = {
        name: run_pipeline(
            name, _hybrid_pipeline(analyzer, vector, bm25), dataset, metrics=metrics
        )
        for name, analyzer in analyzers.items()
    }

    # Query-analysis metrics per analyzer (reuses each analyzer's cache). Keep
    # the analyzed cases so the reference analyzer's failures can be listed.
    analyzed_by_analyzer: dict[str, list[AnalyzedCase]] = {
        name: analyze_dataset(analyzer, dataset.cases)
        for name, analyzer in analyzers.items()
    }
    analysis_reports: dict[str, AnalysisReport] = {
        name: build_analysis_report(analyzed)
        for name, analyzed in analyzed_by_analyzer.items()
    }

    # Latency per analyzer.
    latencies: dict[str, list[StageLatency]] = {
        name: run_latencies(run) for name, run in hybrid_runs.items()
    }

    # Retrieval-strategy contribution, measured with the reference analyzer so
    # vector / BM25 / hybrid see identical query understanding.
    reference_name = REFERENCE_ANALYZER if not rule_only else "Rule-based"
    reference_analyzer = analyzers[reference_name]
    vector_run = run_pipeline(
        "Vector",
        _single_pipeline(reference_analyzer, "Vector", vector),
        dataset,
        metrics=metrics,
    )
    bm25_run = run_pipeline(
        "BM25",
        _single_pipeline(reference_analyzer, "BM25", bm25),
        dataset,
        metrics=metrics,
    )
    contribution = hybrid_contribution(
        vector_run.report.evaluations,
        bm25_run.report.evaluations,
        hybrid_runs[reference_name].report.evaluations,
        k=PRIMARY_K,
    )

    # Optional reranking pass on the reference analyzer for reranker diagnostics.
    rerank_run: PipelineRun | None = None
    if with_rerank and settings.RERANK_ENABLED:
        reranker = CrossEncoderReranker(
            settings.RERANK_MODEL,
            candidate_pool=settings.RERANK_CANDIDATE_POOL,
            top_n=settings.RERANK_TOP_N,
        )
        rerank_run = run_pipeline(
            "Hybrid+Reranker",
            _hybrid_pipeline(reference_analyzer, vector, bm25, reranker=reranker),
            dataset,
            metrics=metrics,
        )

    _print_report(
        dataset_path=dataset_path,
        dataset=dataset,
        indexed=indexed,
        metrics=metrics,
        hybrid_runs=hybrid_runs,
        analysis_reports=analysis_reports,
        analyzed_by_analyzer=analyzed_by_analyzer,
        contribution=contribution,
        latencies=latencies,
        reference_name=reference_name,
        rerank_run=rerank_run,
        markdown=markdown,
    )


def _print_report(
    *,
    dataset_path: Path,
    dataset: EvaluationDataset,
    indexed: int,
    metrics,
    hybrid_runs: dict[str, PipelineRun],
    analysis_reports: dict[str, AnalysisReport],
    analyzed_by_analyzer: dict[str, list[AnalyzedCase]],
    contribution,
    latencies: dict[str, list[StageLatency]],
    reference_name: str,
    rerank_run: PipelineRun | None,
    markdown: bool,
) -> None:
    analyzer_names = list(hybrid_runs.keys())
    reference = hybrid_runs[reference_name]

    print(pr._SEP)
    print("VAULT - PRODUCTION RETRIEVAL BENCHMARK  (analyzer comparison)")
    print(pr._SEP)
    print(f"Dataset:  {dataset_path.name}  ({len(dataset)} queries)")
    print(f"BM25 index: {indexed} chunks")
    print(f"Analyzers:  {', '.join(analyzer_names)}")
    print(f"Reference:  {reference_name} (strategy + failure analysis)\n")

    reports = {name: run.report for name, run in hybrid_runs.items()}
    print(
        pr.section(
            "1. OVERALL METRICS (per analyzer, delta vs reference)",
            format_overall_comparison(
                reports, target=reference_name, markdown=markdown
            ),
        )
    )
    print(
        pr.section(
            "2a. BY CATEGORY (per analyzer)",
            pr.format_grouped_comparison(
                hybrid_runs, metrics, "category", markdown=markdown
            ),
        )
    )
    print(
        pr.section(
            "2b. BY DIFFICULTY (per analyzer)",
            pr.format_grouped_comparison(
                hybrid_runs, metrics, "difficulty", markdown=markdown
            ),
        )
    )
    print(
        pr.section(
            "3a. QUERY ANALYSIS METRICS (per analyzer)",
            pr.format_analysis_comparison(analysis_reports, markdown=markdown),
        )
    )
    print(
        pr.section(
            f"3b. HYBRID CONTRIBUTION ({reference_name})",
            pr.format_hybrid_contribution(contribution, markdown=markdown),
        )
    )
    print(
        pr.section(
            "4. LATENCY (per analyzer)",
            pr.format_latency_comparison(latencies, markdown=markdown),
        )
    )

    # Section 5: what each analyzer upgrade changes, then reference failures.
    blocks: list[str] = []
    baseline = hybrid_runs.get("Rule-based")
    if baseline is not None:
        for name in analyzer_names:
            if name == "Rule-based":
                continue
            blocks.append(
                pr.format_improvement_over_baseline(
                    baseline, hybrid_runs[name], primary_k=PRIMARY_K
                )
            )
            blocks.append("")

    blocks.append(f"--- Detailed failures for {reference_name} ---")
    blocks.append(pr.format_retrieval_failures(reference, primary_k=PRIMARY_K))
    blocks.append("")
    blocks.append(pr.format_analyzer_failures(analyzed_by_analyzer[reference_name]))
    if rerank_run is not None:
        blocks.append("")
        blocks.append(pr.format_reranker_failures(reference, rerank_run))

    print(pr.section("5. IMPROVEMENT & FAILURE ANALYSIS", "\n".join(blocks)))


if __name__ == "__main__":
    args = sys.argv[1:]
    want_markdown = "--markdown" in args
    rule_only_flag = "--rule-only" in args
    want_rerank = "--rerank" in args
    positional = [a for a in args if not a.startswith("--")]
    path = Path(positional[0]) if positional else DEFAULT_DATASET_PATH
    main(
        path,
        rule_only=rule_only_flag,
        with_rerank=want_rerank,
        markdown=want_markdown,
    )
