"""Group the LLM analyzer's retrieval regressions by failure type.

A *retrieval regression* is a query the deterministic rule-based analyzer gets
right (a relevant document in the top ``PRIMARY_K``) but the LLM analyzer gets
wrong once its produced ``SearchRequest`` flows through the identical hybrid
retriever. This report ranks the worst such regressions and buckets each into a
diagnosable failure type so the fix lands in the analyzer, not the benchmark:

* **Date mismatch** — a date-field filter whose resolved day/operator differs
  from ground truth (e.g. timezone-shifted timestamps, datetime vs date).
* **Unnecessary filtering** — the query needed no filter (lexical/semantic) but
  the LLM invented one, over-constraining retrieval.
* **Enum / value mismatch** — right field, wrong value: a paraphrased or
  mis-cased value that matches no stored metadata.
* **Wrong-field hallucination** — a filter on a field the query never implied.
* **Missing filter** — an expected filter the LLM failed to produce.
* **Semantic query drift** — filters were fine, but the rewritten semantic query
  lost the signal the vector search needed.

The LLM's produced requests are cached to disk (keyed by query) so the expensive
analysis pass runs only once; later runs reclassify instantly.

Run:  python -m app.evaluation.llm_failure_report [dataset.json] [--limit N] [--refresh]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.evaluation.analysis_metrics import _normalize_value
from app.evaluation.dataset import DATE_FIELDS, EvaluationCase, EvaluationDataset
from app.evaluation.metrics import default_metrics
from app.evaluation.pipeline import InstrumentedPipeline, run_pipeline
from app.models.filter import Filter, Operator
from app.models.search_request import SearchRequest
from app.models.search_result import SearchResult
from app.processors.llm_intent_analyzer import LLMIntentAnalyzer
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.query_analyzer import QueryAnalyzer, RuleBasedQueryAnalyzer
from app.processors.schema_discovery import (
    enrich_schema_with_values,
    schema_from_indexed_fields,
)
from app.search import (
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
CACHE_PATH = Path(__file__).parent / "data" / "llm_requests_cache.json"

RETRIEVAL_DEPTH = 10
PRIMARY_K = 5

# Failure-type labels, in the order they are diagnosed (most specific first).
DATE_MISMATCH = "Date mismatch"
UNNECESSARY = "Unnecessary filtering (hallucinated on lexical/semantic query)"
ENUM_MISMATCH = "Enum / value mismatch"
WRONG_FIELD = "Wrong-field hallucination"
MISSING = "Missing filter (under-analysis)"
SEMANTIC_DRIFT = "Semantic query drift / other"

_TYPE_ORDER = [
    UNNECESSARY,
    ENUM_MISMATCH,
    DATE_MISMATCH,
    WRONG_FIELD,
    MISSING,
    SEMANTIC_DRIFT,
]


# ---------------------------------------------------------------------------
# Disk-cached LLM analyzer
# ---------------------------------------------------------------------------
def _request_to_json(request: SearchRequest) -> dict:
    return {
        "semantic_query": request.semantic_query,
        "top_k": request.top_k,
        "filters": [
            {"field": f.field, "operator": f.operator.value, "value": f.value}
            for f in request.filters
        ],
    }


def _request_from_json(data: dict) -> SearchRequest:
    filters = [
        Filter(
            field=f["field"],
            operator=Operator(f["operator"]),
            value=f["value"],
        )
        for f in data.get("filters", [])
    ]
    return SearchRequest(
        semantic_query=data.get("semantic_query", ""),
        filters=filters,
        top_k=data.get("top_k", RETRIEVAL_DEPTH),
    )


class _DiskCachedAnalyzer(QueryAnalyzer):
    """Wrap an analyzer, persisting each query's SearchRequest to a JSON cache.

    The wrapped (LLM) analyzer is invoked at most once per unique query across
    all runs; on failure it degrades to a plain semantic request so a single
    outage never aborts the report.
    """

    def __init__(
        self,
        inner: QueryAnalyzer,
        cache_path: Path,
        *,
        default_top_k: int,
        refresh: bool,
    ) -> None:
        self._inner = inner
        self._path = cache_path
        self._default_top_k = default_top_k
        self._cache: dict[str, SearchRequest] = {}
        if cache_path.exists() and not refresh:
            raw = json.loads(cache_path.read_text())
            self._cache = {q: _request_from_json(d) for q, d in raw.items()}
        self._loaded_keys = set(self._cache)

    def analyze(self, query: str) -> SearchRequest:
        if query not in self._cache:
            try:
                self._cache[query] = self._inner.analyze(query)
            except Exception:
                self._cache[query] = SearchRequest(
                    semantic_query=query, filters=[], top_k=self._default_top_k
                )
            self._flush()
        return self._cache[query]

    def _flush(self) -> None:
        payload = {q: _request_to_json(r) for q, r in self._cache.items()}
        self._path.write_text(json.dumps(payload, indent=2, default=str))

    @property
    def newly_computed(self) -> int:
        return len(set(self._cache) - self._loaded_keys)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _values_match(field: str, a: object, b: object) -> bool:
    return _normalize_value(field, a) == _normalize_value(field, b)


def classify_failure(case: EvaluationCase, produced: tuple[Filter, ...]) -> str:
    """Diagnose why the LLM analyzer's request regresses retrieval for a case."""
    expected = case.expected_filters
    expected_by_field = {f.field: f for f in expected}
    produced_by_field = {f.field: f for f in produced}

    # 1. Date mismatch — a produced date filter that disagrees with ground truth,
    #    or a temporal case whose expected date filter the LLM mangled.
    for f in produced:
        if f.field in DATE_FIELDS:
            exp = expected_by_field.get(f.field)
            if (
                exp is None
                or exp.operator != f.operator
                or not _values_match(f.field, exp.value, f.value)
            ):
                return DATE_MISMATCH
    if any(f.field in DATE_FIELDS for f in expected) and not any(
        f.field in DATE_FIELDS for f in produced
    ):
        return DATE_MISMATCH

    # 2. Unnecessary filtering — the query needed no filter but the LLM added one.
    if not expected and produced:
        return UNNECESSARY

    # 3. Enum / value mismatch — right field, wrong (paraphrased/mis-cased) value.
    for f in produced:
        exp = expected_by_field.get(f.field)
        if exp is not None and not _values_match(f.field, exp.value, f.value):
            return ENUM_MISMATCH

    # 4. Wrong-field hallucination — a filter on a field the query never implied.
    if expected and any(f.field not in expected_by_field for f in produced):
        return WRONG_FIELD

    # 5. Missing filter — an expected filter the LLM never produced.
    if any(f.field not in produced_by_field for f in expected):
        return MISSING

    # 6. Otherwise the filters were fine; the semantic rewrite must have drifted.
    return SEMANTIC_DRIFT


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------
@dataclass
class Regression:
    case: EvaluationCase
    rule_rank: int | None
    llm_rank: int | None
    rule_request: SearchRequest
    llm_request: SearchRequest
    failure_type: str

    @property
    def severity(self) -> tuple:
        # Worst first: LLM never retrieved the doc, then larger LLM rank, then the
        # better the baseline rank (bigger drop) the worse the regression.
        llm_missing = 0 if self.llm_rank is None else 1
        return (llm_missing, self.llm_rank or 0, self.rule_rank or 0)


def _first_relevant_rank(
    results: list[SearchResult], expected_docs: tuple[str, ...]
) -> int | None:
    expected = set(expected_docs)
    for rank, result in enumerate(results, start=1):
        if result.chunk.document_id in expected:
            return rank
    return None


def _filters_str(filters) -> str:
    if not filters:
        return "(none)"
    return "; ".join(
        f"{f.field} {f.operator.value} {f.value!r}" for f in filters
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
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


def _hybrid(analyzer, vector, bm25) -> InstrumentedPipeline:
    return InstrumentedPipeline(
        analyzer,
        [("Vector", vector), ("BM25", bm25)],
        ReciprocalRankFusion(k=settings.RRF_K),
    )


def collect_regressions(
    dataset: EvaluationDataset,
    *,
    refresh: bool,
) -> tuple[list[Regression], int]:
    qdrant = QdrantService(get_qdrant_client())
    fields, multi = _discover(qdrant)
    values = qdrant.discover_property_values()

    index = RankBM25Index()
    BM25Indexer(qdrant_service=qdrant, index=index).rebuild()

    registry = _registry(fields, multi)
    schema = _load_schema(fields, multi, values)

    rule = RuleBasedQueryAnalyzer(registry=registry, default_top_k=RETRIEVAL_DEPTH)
    llm = _DiskCachedAnalyzer(
        LLMIntentAnalyzer(build_intent_llm(), schema, default_top_k=RETRIEVAL_DEPTH),
        CACHE_PATH,
        default_top_k=RETRIEVAL_DEPTH,
        refresh=refresh,
    )

    vector = VectorSearchStrategy(
        embedding_service=EmbeddingService(), qdrant_service=qdrant
    )
    bm25 = BM25SearchStrategy(index=index)

    metrics = default_metrics()
    rule_run = run_pipeline(
        "Rule-based", _hybrid(rule, vector, bm25), dataset, metrics=metrics
    )
    llm_run = run_pipeline(
        "LLM-only", _hybrid(llm, vector, bm25), dataset, metrics=metrics
    )

    regressions: list[Regression] = []
    for case in dataset.cases:
        if not case.expected_documents:
            continue
        rule_rank = _first_relevant_rank(
            rule_run.results_by_case[case.id], case.expected_documents
        )
        llm_rank = _first_relevant_rank(
            llm_run.results_by_case[case.id], case.expected_documents
        )
        rule_hit = rule_rank is not None and rule_rank <= PRIMARY_K
        llm_hit = llm_rank is not None and llm_rank <= PRIMARY_K
        if rule_hit and not llm_hit:
            llm_request = llm.analyze(case.query)
            regressions.append(
                Regression(
                    case=case,
                    rule_rank=rule_rank,
                    llm_rank=llm_rank,
                    rule_request=rule.analyze(case.query),
                    llm_request=llm_request,
                    failure_type=classify_failure(
                        case, tuple(llm_request.filters)
                    ),
                )
            )

    regressions.sort(key=lambda r: r.severity)
    return regressions, llm.newly_computed


def format_report(regressions: list[Regression], *, limit: int) -> str:
    top = regressions[:limit]
    grouped: dict[str, list[Regression]] = defaultdict(list)
    for reg in top:
        grouped[reg.failure_type].append(reg)

    sep = "=" * 92
    lines: list[str] = [
        sep,
        f"LLM ANALYZER RETRIEVAL REGRESSIONS  (rule-based hit@{PRIMARY_K}, "
        f"LLM miss@{PRIMARY_K})",
        sep,
        f"Total regressions found: {len(regressions)}   "
        f"Reported (top by severity): {len(top)}",
        "",
        "Breakdown by failure type:",
    ]
    for failure_type in _TYPE_ORDER:
        count = len(grouped.get(failure_type, []))
        if count:
            lines.append(f"  {count:>3}  {failure_type}")
    lines.append("")

    for failure_type in _TYPE_ORDER:
        bucket = grouped.get(failure_type)
        if not bucket:
            continue
        lines.append("-" * 92)
        lines.append(f"{failure_type}  ({len(bucket)})")
        lines.append("-" * 92)
        for reg in bucket:
            rule_pos = "not retrieved" if reg.rule_rank is None else f"rank {reg.rule_rank}"
            llm_pos = "not retrieved" if reg.llm_rank is None else f"rank {reg.llm_rank}"
            lines.append(f"[{reg.case.category or '-'}] {reg.case.id}")
            lines.append(f"    query    : {reg.case.query}")
            lines.append(f"    retrieval: rule {rule_pos}  ->  LLM {llm_pos}")
            lines.append(
                f"    expected : {_filters_str(reg.case.expected_filters)}"
            )
            lines.append(
                f"    produced : {_filters_str(reg.llm_request.filters)}"
            )
            if reg.rule_request.semantic_query != reg.llm_request.semantic_query:
                lines.append(
                    f"    semantic : {reg.rule_request.semantic_query!r}  ->  "
                    f"{reg.llm_request.semantic_query!r}"
                )
        lines.append("")

    return "\n".join(lines)


def main(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    limit: int = 50,
    refresh: bool = False,
) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)
    regressions, newly = collect_regressions(dataset, refresh=refresh)
    if newly:
        print(f"(computed {newly} new LLM analyses; cached to {CACHE_PATH.name})\n")
    print(format_report(regressions, limit=limit))


if __name__ == "__main__":
    args = sys.argv[1:]
    refresh_flag = "--refresh" in args
    limit_value = 50
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args):
            limit_value = int(args[idx + 1])
    positional = [
        a
        for i, a in enumerate(args)
        if not a.startswith("--") and (i == 0 or args[i - 1] != "--limit")
    ]
    path = Path(positional[0]) if positional else DEFAULT_DATASET_PATH
    main(path, limit=limit_value, refresh=refresh_flag)
