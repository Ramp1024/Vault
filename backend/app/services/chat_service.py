from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from app.models.rag_response import RAGResponse
from app.models.search_result import SearchResult
from app.processors.prompt_builder import PromptBuilder
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService


@dataclass(frozen=True)
class ChatContext:
    query: str
    sources: list[SearchResult]
    prompt: str


class ChatService:
    """Orchestrates query embedding, retrieval, prompt construction, and generation."""

    RETRIEVAL_LIMIT = 5

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_service: QdrantService | None = None,
        prompt_builder: PromptBuilder | None = None,
        generation_service: GenerationService | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService(get_qdrant_client())
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generation_service = generation_service or GenerationService()

    def retrieve_sources(self, query: str) -> list[SearchResult]:
        query_embedding = self.embedding_service.embed(query)
        return self.qdrant_service.search(
            query_embedding=query_embedding,
            limit=self.RETRIEVAL_LIMIT,
        )

    def build_prompt(self, query: str, sources: list[SearchResult]) -> str:
        return self.prompt_builder.build(query=query, results=sources)

    def prepare(self, query: str) -> ChatContext:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        sources = self.retrieve_sources(normalized_query)
        prompt = self.build_prompt(normalized_query, sources)
        return ChatContext(query=normalized_query, sources=sources, prompt=prompt)

    def answer(self, query: str) -> RAGResponse:
        context = self.prepare(query)
        answer = self.generation_service.generate(prompt=context.prompt)
        return RAGResponse(answer=answer, sources=context.sources)

    def stream_answer(
        self, query: str, context: ChatContext | None = None
    ) -> Iterator[str]:
        context = context or self.prepare(query)
        yield from self.generation_service.stream_generate(prompt=context.prompt)


def get_chat_service() -> ChatService:
    return ChatService()


# Backward-compatible aliases: RAGService now maps directly to ChatService.
RAGService = ChatService


def get_rag_service() -> ChatService:
    return get_chat_service()
