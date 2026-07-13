from app.models.search_result import SearchResult


class PromptBuilder:
    """Builds answer prompts from retrieval results without invoking any LLM."""

    def build(self, query: str, results: list[SearchResult]) -> str:
        """Assemble a constrained prompt for retrieval-grounded answering.

        Args:
            query: End-user question.
            results: Retrieved search results to use as context.

        Returns:
            A prompt string ready to send to an LLM.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        context = self._format_context(results)

        return (
            "You are Vault, a retrieval-grounded assistant.\n\n"
            "Use ONLY the supplied context to answer the question.\n"
            "If the answer is not present in the context, explicitly say you don't know.\n\n"
            "Context:\n\n"
            f"{context}\n\n"
            "Question:\n\n"
            f"{normalized_query}\n"
        )

    def _format_context(self, results: list[SearchResult]) -> str:
        """Render retrieved chunks as labeled context blocks."""
        if not results:
            return "[Chunk 1]\nNo context provided."

        sections: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            sections.append(
                "\n".join(
                    [
                        f"[Chunk {index}]",
                        f"Document: {chunk.document_title}",
                        f"Chunk Index: {chunk.chunk_index}",
                        chunk.content,
                    ]
                )
            )

        return "\n\n".join(sections)