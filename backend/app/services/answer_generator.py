from __future__ import annotations

from collections.abc import Iterator

from app.models.answer import GeneratedAnswer
from app.models.context import AssembledContext
from app.models.prompt import Prompt
from app.processors.citation_mapper import CitationMapper
from app.services.llm import LLM


class AnswerGenerator:
    """Orchestrate a single generation step and return a structured answer.

    Given a fully built :class:`Prompt` and the :class:`AssembledContext` it was
    grounded in, the generator invokes the LLM, parses the response, and resolves
    citations. It is deliberately independent of retrieval: it knows nothing about
    search engines, strategies, or query analysis.
    """

    def __init__(
        self,
        llm: LLM,
        citation_mapper: CitationMapper | None = None,
    ) -> None:
        self.llm = llm
        self.citation_mapper = citation_mapper or CitationMapper()

    def generate(self, prompt: Prompt, context: AssembledContext) -> GeneratedAnswer:
        """Generate a structured answer for ``prompt`` grounded in ``context``.

        Args:
            prompt: The system/user prompt produced by the prompt template.
            context: The assembled context whose reference ids the answer cites.

        Returns:
            A :class:`GeneratedAnswer` with the answer text and deterministically
            resolved citations. ``confidence`` is left ``None`` at this stage.
        """
        raw_answer = self.llm.generate(prompt).strip()
        citations = self.citation_mapper.map(raw_answer, context)
        return GeneratedAnswer(answer=raw_answer, citations=citations)

    def stream(self, prompt: Prompt) -> Iterator[str]:
        """Stream raw answer text fragments from the LLM.

        Citations are resolved from the *complete* answer, so streaming yields
        only the raw text; callers that need structured citations should use
        :meth:`generate` or map citations from the accumulated text once the
        stream finishes.
        """
        yield from self.llm.stream(prompt)
