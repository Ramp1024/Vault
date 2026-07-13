from app.models.rag_response import RAGResponse
from app.processors.prompt_builder import PromptBuilder
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService


class RAGService:
    """Coordinates the retrieval-augmented generation pipeline."""

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

    def answer(self, query: str) -> RAGResponse:
        """Run Query -> Embed -> Retrieve -> Build Prompt -> Generate -> Return."""
        query_embedding = self.embedding_service.embed(query)
        sources = self.qdrant_service.search(
            query_embedding=query_embedding,
            limit=self.RETRIEVAL_LIMIT,
        )
        prompt = self.prompt_builder.build(query=query, results=sources)
        answer = self.generation_service.generate(prompt=prompt)

        return RAGResponse(answer=answer, sources=sources)


def get_rag_service() -> RAGService:
    """Factory function to create a RAG service instance."""
    return RAGService()