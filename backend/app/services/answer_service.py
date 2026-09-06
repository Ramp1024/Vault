from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from app.core.config import settings
from app.models.answer import Citation, GeneratedAnswer
from app.models.context import AssembledContext
from app.models.filter import Filter, Operator
from app.models.prompt import Prompt
from app.models.search_result import SearchResult
from app.processors.citation_mapper import CitationMapper
from app.processors.context_builder import ContextBuilder
from app.processors.metadata_registry import (
    MetadataRegistry,
    default_metadata_registry,
)
from app.processors.prompt_template import PromptTemplate, build_prompt_template
from app.processors.query_analyzer import (
    QueryAnalyzer,
    RuleBasedQueryAnalyzer,
)
from app.processors.schema_discovery import (
    enrich_schema_with_values,
    schema_from_indexed_fields,
    with_temporal_metadata,
)
from app.processors.augmenting_analyzer import AugmentingIntentAnalyzer
from app.processors.constraint_proposal import LLMConstraintProposer
from app.processors.constraint_validation import ConstraintValidator
from app.processors.query_intent import DeterministicIntentAnalyzer
from app.models.metadata_schema import MetadataSchema
from app.services.metadata_schema_store import MetadataSchemaStore
from app.search import (
    BM25SearchStrategy,
    CrossEncoderReranker,
    Reranker,
    RetrievalMode,
    SearchEngine,
    VectorSearchStrategy,
    build_search_engine,
)
from app.services.answer_generator import AnswerGenerator
from app.services.embedding_service import EmbeddingService
from app.services.llm import build_intent_llm, build_llm
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService


@dataclass(frozen=True)
class AnswerContext:
    """Prepared generation inputs: the assembled context and the built prompt.

    Lets the API retrieve, build context, and construct the prompt up front (so
    failures surface as proper HTTP errors) before opening a stream, then reuse
    the same prepared state for both the sources frame and generation.
    """

    query: str
    context: AssembledContext
    prompt: Prompt


class AnswerService:
    """End-to-end answer generation on top of the retrieval pipeline.

    Owns the full generation layer — retrieval, context building, prompt
    construction, and structured (or streamed) generation — while keeping each
    stage swappable and independent. Retrieval flows in through the public
    ``SearchEngine.search`` API and never leaks back into generation; the LLM is
    reached only through the ``LLM`` abstraction.
    """

    RETRIEVAL_LIMIT = 5

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        search_engine: SearchEngine | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_template: PromptTemplate | None = None,
        answer_generator: AnswerGenerator | None = None,
        citation_mapper: CitationMapper | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService(get_qdrant_client())
        self.context_builder = context_builder or ContextBuilder(
            token_budget=settings.CONTEXT_TOKEN_BUDGET
        )
        self.prompt_template = prompt_template or build_prompt_template(
            settings.PROMPT_TEMPLATE
        )
        # The citation mapper is a first-class collaborator so both non-streaming
        # generation (via the answer generator) and the streaming path share one
        # authoritative validator. When a custom generator is supplied, adopt its
        # mapper so the service and generator never diverge.
        self.citation_mapper = citation_mapper or (
            answer_generator.citation_mapper
            if answer_generator is not None
            else CitationMapper()
        )
        self.answer_generator = answer_generator or AnswerGenerator(
            build_llm(), citation_mapper=self.citation_mapper
        )

        reranker = self._build_reranker()
        # When reranking is enabled we must retrieve a larger candidate pool for
        # the cross-encoder to reorder; otherwise retrieve just the final count.
        retrieval_depth = (
            settings.RERANK_CANDIDATE_POOL
            if reranker is not None
            else self.RETRIEVAL_LIMIT
        )
        self.query_analyzer = query_analyzer or self._build_query_analyzer(
            retrieval_depth
        )
        self.search_engine = search_engine or build_search_engine(
            self._retrieval_mode(),
            self.query_analyzer,
            vector_strategy=VectorSearchStrategy(
                embedding_service=self.embedding_service,
                qdrant_service=self.qdrant_service,
            ),
            bm25_strategy=BM25SearchStrategy(),
            rrf_k=settings.RRF_K,
            reranker=reranker,
        )

    @staticmethod
    def _build_reranker() -> Reranker | None:
        """Build the cross-encoder reranker when enabled, else ``None``.

        Returning ``None`` keeps the pipeline reranker-free (the model is never
        loaded), so the heavy cross-encoder stack is only touched when
        ``RERANK_ENABLED`` is set.
        """
        if not settings.RERANK_ENABLED:
            return None
        return CrossEncoderReranker(
            settings.RERANK_MODEL,
            candidate_pool=settings.RERANK_CANDIDATE_POOL,
            top_n=settings.RERANK_TOP_N,
        )

    @staticmethod
    def _retrieval_mode() -> RetrievalMode:
        """Resolve the configured retrieval mode (defaults to hybrid)."""
        return RetrievalMode(settings.RETRIEVAL_MODE.strip().lower())

    def _build_registry(self) -> MetadataRegistry:
        """Derive the metadata registry from indexed property names.

        Falls back to the connector default when nothing can be discovered
        (e.g. an empty or unreachable collection), so recognized filter fields
        track what is actually indexed instead of a hardcoded list.
        """
        try:
            fields, multi_fields = self.qdrant_service.discover_property_fields()
        except Exception:
            fields, multi_fields = [], set()

        if not fields:
            return default_metadata_registry()
        return MetadataRegistry.from_indexed_fields(fields, multi_fields)

    def _build_query_analyzer(self, retrieval_depth: int) -> QueryAnalyzer:
        """Build the query analyzer used to turn a query into a SearchRequest.

        When a metadata schema is available the default is the deterministic,
        schema-aware :class:`DeterministicIntentAnalyzer` (lexical routing, value
        validated metadata filters, and schema-driven temporal ranges — no LLM).
        With ``INTENT_ANALYZER_ENABLED`` it is wrapped in an
        :class:`AugmentingIntentAnalyzer`: the deterministic result stays
        authoritative and the LLM may only add validated, grounded constraints
        for fields the deterministic stage left unclaimed — it can never override
        a trusted filter or rewrite the subject. Without a schema it falls back to
        the rule-based analyzer. The retrieval engine stays unaware of which
        analyzer produced the ``SearchRequest``.
        """
        rule_based = RuleBasedQueryAnalyzer(
            registry=self._build_registry(),
            default_top_k=retrieval_depth,
        )

        schema = self._load_metadata_schema()
        if not schema:
            return rule_based

        deterministic = DeterministicIntentAnalyzer(
            schema, default_top_k=retrieval_depth
        )
        if not settings.INTENT_ANALYZER_ENABLED:
            return deterministic

        return AugmentingIntentAnalyzer(
            deterministic=deterministic,
            proposer=LLMConstraintProposer(build_intent_llm(), schema),
            validator=ConstraintValidator(
                schema,
                min_confidence=settings.INTENT_LLM_MIN_CONFIDENCE,
                min_grounding=settings.INTENT_GROUNDING_THRESHOLD,
            ),
        )

    def _load_metadata_schema(self) -> MetadataSchema:
        """Load the schema and ensure value + temporal enrichment.

        The persisted schema may predate value/temporal-role enrichment, so
        observed field values are folded in (populating ``allowed_values`` and
        temporal roles). This keeps the live analyzer aligned with discovery even
        when the on-disk schema is stale; if no values can be read, temporal
        metadata is still applied so date fields remain query-selectable.
        """
        schema = MetadataSchemaStore().load()
        if not schema:
            try:
                fields, multi_fields = self.qdrant_service.discover_property_fields()
            except Exception:
                fields, multi_fields = [], set()
            schema = schema_from_indexed_fields(fields, multi_fields)

        try:
            values = self.qdrant_service.discover_property_values()
        except Exception:
            values = {}
        return (
            enrich_schema_with_values(schema, values)
            if values
            else with_temporal_metadata(schema)
        )

    def retrieve(self, query: str) -> list[SearchResult]:
        """Run retrieval and cap to the configured number of sources."""
        # Hybrid fusion can surface more than RETRIEVAL_LIMIT chunks (each
        # strategy contributes candidates); cap to the configured limit so the
        # context receives a consistent number of sources regardless of mode.
        return self.search_engine.search(query)[: self.RETRIEVAL_LIMIT]

    def build_context(self, results: list[SearchResult]) -> AssembledContext:
        """Assemble reranked results into an LLM-ready context."""
        return self.context_builder.build(results)

    def build_prompt(self, query: str, context: AssembledContext) -> Prompt:
        """Construct the prompt for ``query`` grounded in ``context``."""
        return self.prompt_template.build(query, context)

    def generate(self, prompt: Prompt, context: AssembledContext) -> GeneratedAnswer:
        """Invoke the LLM and return a structured, cited answer."""
        return self.answer_generator.generate(prompt, context)

    def map_citations(
        self, answer_text: str, context: AssembledContext
    ) -> tuple[Citation, ...]:
        """Validate model output against ``context`` into grounded citations.

        The single authoritative entry point used by the streaming endpoint,
        which maps citations once the full answer text has been received.
        """
        return self.citation_mapper.map(answer_text, context)

    def prepare(self, query: str) -> AnswerContext:
        """Retrieve, build context, and construct the prompt without generating.

        Raises:
            ValueError: When ``query`` is empty.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        outcome = self.search_engine.retrieve(normalized_query)
        context = self.build_context(outcome.results[: self.RETRIEVAL_LIMIT])
        notice = self._relaxation_notice(outcome.relaxed_filters)
        if notice is not None:
            context = replace(context, notice=notice)
        prompt = self.build_prompt(normalized_query, context)
        return AnswerContext(query=normalized_query, context=context, prompt=prompt)

    @staticmethod
    def _relaxation_notice(relaxed_filters: tuple[Filter, ...]) -> str | None:
        """Warn generation when a requested date filter matched nothing.

        Fail-open drops the filter and returns semantically similar sources that
        may be from other dates; without this warning the model tends to relabel
        them with the asked-for date. Only date ranges are surfaced — other
        relaxed filters do not risk the same date-attribution error.
        """
        asked: list[str] = []
        for f in relaxed_filters:
            if f.operator is not Operator.BETWEEN:
                continue
            if not isinstance(f.value, (list, tuple)) or len(f.value) != 2:
                continue
            low, high = f.value
            asked.append(str(low) if low == high else f"{low} to {high}")
        if not asked:
            return None
        dates = ", ".join(asked)
        return (
            f"No stored entry matches the date(s) requested ({dates}). The sources "
            f"below were retrieved by semantic relevance and may be from other "
            f"dates; do not claim they are from {dates}."
        )

    def answer(self, query: str) -> GeneratedAnswer:
        """Run the full generation pipeline and return a structured answer."""
        prepared = self.prepare(query)
        return self.generate(prepared.prompt, prepared.context)

    def stream_answer(
        self, query: str, context: AnswerContext | None = None
    ) -> Iterator[str]:
        """Stream raw answer text for ``query``.

        Accepts a pre-built :class:`AnswerContext` so the API can prepare (and
        surface preparation errors) before opening the stream.
        """
        prepared = context or self.prepare(query)
        yield from self.answer_generator.stream(prepared.prompt)


def get_answer_service() -> AnswerService:
    """Factory returning a fully wired answer service."""
    return AnswerService()
