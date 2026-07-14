"""RAG verification script.

Verifies end-to-end behavior:
1. Query -> Embed -> Retrieve -> Build Prompt -> Generate -> Return
2. Answer is grounded in retrieved chunks (lexical-overlap heuristic)
3. Sources are returned
4. Unknown topic responses are honest and non-fabricated
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

KNOWN_QUERIES = [
    "What is Webpack?",
    "Explain Apache Arrow.",
    "What exercises do I perform on back day?",
    "What is Docker?",
]

UNKNOWN_QUERY = "What is Tree-sitter?"
UNKNOWN_EXPECTED_PHRASE = (
    "I couldn't find relevant information in your knowledge base."
)

MIN_GROUNDING_RATIO = 0.20
MAX_UNMATCHED_TERMS = 20
MAX_PROMPT_CHARS = 50_000
MAX_CONTEXT_WORDS = 3_000

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
    "you",
    "your",
}


class VerificationFailure(Exception):
    """Raised when a RAG verification check fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def _normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower())
    return {t for t in tokens if len(t) >= 4 and t not in STOPWORDS}


def _ensure_index(qdrant, embedding_service) -> None:
    from app.connectors.notion.connector import NotionConnector
    from app.processors.chunker import Chunker

    if qdrant.collection_exists() and qdrant.count() > 0:
        return

    print("Indexing chunks because collection is empty...")
    documents = NotionConnector().ingest()
    chunks = Chunker().chunk_documents(documents)
    embedded_chunks = embedding_service.embed_chunks(chunks)
    qdrant.upsert(embedded_chunks)


def _verify_grounding(answer: str, sources_text: str) -> tuple[float, int]:
    answer_terms = _normalize_tokens(answer)
    source_terms = _normalize_tokens(sources_text)

    if not answer_terms:
        return 0.0, 0

    overlap = answer_terms & source_terms
    unmatched = answer_terms - source_terms
    grounding_ratio = len(overlap) / len(answer_terms)
    return grounding_ratio, len(unmatched)


def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _print_prompt_stats(prompt: str, sources_count: int, context_word_count: int) -> None:
    approx_tokens = _approx_token_count(prompt)

    print(f"  Sources used      : {sources_count}")
    print(f"  Context words     : {context_word_count}")
    print(f"  Prompt chars      : {len(prompt)}")
    print(f"  Approx tokens     : {approx_tokens}")
    print(prompt[:500])

    _require(
        len(prompt) < MAX_PROMPT_CHARS,
        f"Prompt is too large: {len(prompt)} chars (max {MAX_PROMPT_CHARS})",
    )
    _require(
        context_word_count <= MAX_CONTEXT_WORDS,
        f"Context is too large: {context_word_count} words "
        f"(max {MAX_CONTEXT_WORDS})",
    )


def verify_rag() -> None:
    from app.processors.prompt_builder import PromptBuilder
    from app.services.embedding_service import EmbeddingService
    from app.services.qdrant import get_qdrant_client
    from app.services.qdrant_service import QdrantService
    from app.services.rag_service import RAGService

    print("=" * 90)
    print("VAULT - RAG VERIFICATION")
    print("=" * 90)

    embedding_service = EmbeddingService()
    qdrant = QdrantService(client=get_qdrant_client())
    prompt_builder = PromptBuilder()
    rag = RAGService(
        embedding_service=embedding_service,
        qdrant_service=qdrant,
        prompt_builder=prompt_builder,
    )

    _ensure_index(qdrant, embedding_service)

    for query in KNOWN_QUERIES:
        print(f"\nQuery: {query}")
        query_embedding = embedding_service.embed(query)
        sources = qdrant.search(query_embedding, limit=rag.RETRIEVAL_LIMIT)
        prompt = prompt_builder.build(query, sources)
        context_word_count = sum(len(item.chunk.content.split()) for item in sources)

        _print_prompt_stats(prompt, len(sources), context_word_count)

        response = rag.answer(query)

        _require(response.answer.strip(), f"Empty answer returned for query: {query}")
        _require(response.sources, f"No sources returned for query: {query}")

        sources_text = "\n\n".join(item.chunk.content for item in response.sources)
        grounding_ratio, unmatched_count = _verify_grounding(response.answer, sources_text)

        _require(
            grounding_ratio >= MIN_GROUNDING_RATIO,
            f"Answer appears weakly grounded for query '{query}'. "
            f"grounding_ratio={grounding_ratio:.2f}, "
            f"required>={MIN_GROUNDING_RATIO:.2f}",
        )
        _require(
            unmatched_count <= MAX_UNMATCHED_TERMS,
            f"Potential hallucination risk for query '{query}'. "
            f"unmatched_terms={unmatched_count}, max={MAX_UNMATCHED_TERMS}",
        )

        print(f"  Sources returned : {len(response.sources)}")
        print(f"  Grounding ratio  : {grounding_ratio:.2f}")
        print(f"  Unmatched terms  : {unmatched_count}")
        print(f"  Answer preview   : {response.answer[:200].replace(chr(10), ' ')}")

    print(f"\nUnknown query: {UNKNOWN_QUERY}")
    unknown_query_embedding = embedding_service.embed(UNKNOWN_QUERY)
    unknown_sources = qdrant.search(
        unknown_query_embedding,
        limit=rag.RETRIEVAL_LIMIT,
    )
    unknown_prompt = prompt_builder.build(UNKNOWN_QUERY, unknown_sources)
    unknown_context_word_count = sum(
        len(item.chunk.content.split()) for item in unknown_sources
    )

    _print_prompt_stats(
        unknown_prompt,
        len(unknown_sources),
        unknown_context_word_count,
    )

    unknown_response = rag.answer(UNKNOWN_QUERY)

    print(f"  Sources returned : {len(unknown_response.sources)}")
    print(f"  Answer           : {unknown_response.answer}")

    _require(
        unknown_response.sources,
        "Unknown-topic query returned no sources list",
    )
    _require(
        UNKNOWN_EXPECTED_PHRASE.lower() in unknown_response.answer.lower(),
        "Unknown-topic answer did not include the required honest fallback phrase: "
        f"'{UNKNOWN_EXPECTED_PHRASE}'",
    )

    print("\n✅ RAG verification passed")


if __name__ == "__main__":
    try:
        verify_rag()
    except VerificationFailure as exc:
        print(f"\n❌ RAG verification failed: {exc}")
        raise SystemExit(1)