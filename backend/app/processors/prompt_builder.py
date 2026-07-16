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
            "You are Vault, a helpful assistant answering from the user's "
            "knowledge base.\n\n"
            "Answer the user's question directly using only the supplied context.\n"
            "Synthesize related facts into a natural response instead of describing "
            "or enumerating the context passages.\n"
            "Do not mention chunks, chunk numbers, document labels, supplied context, "
            "retrieval, or the knowledge base in the answer.\n"
            "Do not add an offer to clarify, expand, or answer more questions.\n"
            "Use concise prose or bullets according to what best fits the question.\n"
            "If the context does not contain enough information to answer, respond "
            "exactly: I couldn't find relevant information in your knowledge base.\n\n"
            "Internal context (never refer to these labels in the answer):\n\n"
            f"{context}\n\n"
            "User question:\n\n"
            f"{normalized_query}\n\n"
            "Answer:\n"
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