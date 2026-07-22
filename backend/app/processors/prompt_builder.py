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
            lines = [
                f"[Chunk {index}]",
                f"Document: {chunk.document_title}",
                f"Chunk Index: {chunk.chunk_index}",
            ]

            properties = self._format_properties(chunk.metadata.get("properties"))
            if properties:
                lines.append(properties)

            lines.append(chunk.content)
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _format_properties(self, properties: object) -> str:
        """Render a chunk's structured properties as readable context lines."""
        if not isinstance(properties, dict) or not properties:
            return ""

        lines: list[str] = ["Properties:"]
        for name, value in properties.items():
            if isinstance(value, (list, tuple)):
                rendered = ", ".join(str(item) for item in value)
            else:
                rendered = str(value)
            lines.append(f"  {name}: {rendered}")

        return "\n".join(lines)