from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from app.core.config import settings
from app.models.answer import Citation, GeneratedAnswer
from app.models.context import AssembledContext
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
    CompositeQueryAnalyzer,
    QueryAnalyzer,
    RuleBasedQueryAnalyzer,
)
from app.processors.schema_discovery import schema_from_indexed_fields
from app.processors.llm_intent_analyzer import LLMIntentAnalyzer
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
        """Build the query analyzer, optionally composing rule-based + LLM intent.

        The deterministic rule-based analyzer is always present. When
        ``INTENT_ANALYZER_ENABLED`` is set and a metadata schema is available, it
        is wrapped in a :class:`CompositeQueryAnalyzer` that also consults the
        schema-aware LLM intent analyzer (which owns authoring-time date routing
        via its temporal descriptor). The retrieval engine remains unaware of
        which analyzer produced the ``SearchRequest``.
        """
        rule_based = RuleBasedQueryAnalyzer(
            registry=self._build_registry(),
            default_top_k=retrieval_depth,
        )

        if not settings.INTENT_ANALYZER_ENABLED:
            return rule_based

        schema = self._load_metadata_schema()
        if not schema:
            return rule_based

        llm_analyzer = LLMIntentAnalyzer(
            build_intent_llm(),
            schema,
            default_top_k=retrieval_depth,
        )
        return CompositeQueryAnalyzer(rule_based=rule_based, llm_based=llm_analyzer)

    def _load_metadata_schema(self) -> MetadataSchema:
        """Load the persisted schema, falling back to indexed field discovery."""
        schema = MetadataSchemaStore().load()
        if schema:
            return schema
        try:
            fields, multi_fields = self.qdrant_service.discover_property_fields()
        except Exception:
            fields, multi_fields = [], set()
        return schema_from_indexed_fields(fields, multi_fields)

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

        results = self.retrieve(normalized_query)
        context = self.build_context(results)
        prompt = self.build_prompt(normalized_query, context)
        return AnswerContext(query=normalized_query, context=context, prompt=prompt)

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
