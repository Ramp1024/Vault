from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from app.models.context import AssembledContext
from app.models.prompt import Prompt


class PromptTemplate(ABC):
    """Backend-independent prompt construction from an assembled context.

    Isolates *what* the LLM is told (system rules, question framing, context
    rendering, citation instructions) from *how* it is generated. Templates are
    swappable so prompting can evolve without touching retrieval, context
    building, or the LLM abstraction.
    """

    @abstractmethod
    def build(self, query: str, context: AssembledContext) -> Prompt:
        """Return a :class:`Prompt` for ``query`` grounded in ``context``."""
        raise NotImplementedError


class GroundedAnswerTemplate(PromptTemplate):
    """Default template: answer strictly from context and cite reference ids.

    The prompt instructs the model to cite the numeric reference ids assigned by
    the context builder (``[1]``, ``[2]``) rather than inventing sources, which
    keeps citations grounded in retrieved content.
    """

    _NO_ANSWER = "I couldn't find relevant information in your knowledge base."

    _SYSTEM = (
        "You are Vault, a helpful assistant answering from the user's knowledge "
        "base.\n"
        "Answer the question using only the numbered sources provided.\n"
        "Synthesize related facts into a natural response instead of describing "
        "or enumerating the sources.\n"
        "After each sentence or claim, cite the sources it came from using their "
        "reference numbers in square brackets, e.g. [1] or [2][3].\n"
        "Only cite reference numbers that appear in the provided sources; never "
        "invent citations or facts.\n"
        "Do not add an offer to clarify, expand, or answer more questions.\n"
        "State a date or day of the week only when a source's own properties give "
        "it; never infer a source's date from the question, and never relabel a "
        "source with a date or weekday other than the one its properties show.\n"
        "When the question specifies a constraint such as a date, time, or "
        "category, treat a source as satisfying it only when the source's own "
        "properties show it does; summarize that source's content even if its "
        "topic wording differs, but never assume an unrelated source satisfies "
        "the constraint.\n"
        "If the question asks about a specific date and no source's properties "
        "match that date, say you couldn't find an entry for that date rather "
        "than attributing another source's content to it.\n"
        f"Only when no provided source is relevant to the question, respond "
        f"exactly: {_NO_ANSWER}"
    )

    def build(self, query: str, context: AssembledContext) -> Prompt:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        sections = ["Sources:\n\n", f"{self._format_context(context)}\n\n"]
        if context.notice:
            sections.append(f"Note: {context.notice}\n\n")
        sections.append("Question:\n\n")
        sections.append(f"{normalized_query}\n\n")
        sections.append("Answer (with bracketed citations):")
        return Prompt(system=self._SYSTEM, user="".join(sections))

    def _format_context(self, context: AssembledContext) -> str:
        """Render each context chunk as a numbered, citable source block."""
        if context.is_empty():
            return "No sources were retrieved."

        blocks: list[str] = []
        for item in context.chunks:
            chunk = item.chunk
            lines = [
                f"[{item.reference_id}] {chunk.document_title}",
            ]
            properties = self._format_properties(chunk.metadata.get("properties"))
            if properties:
                lines.append(properties)
            lines.append(chunk.content)
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    @staticmethod
    def _format_properties(properties: Any) -> str:
        """Render structured chunk properties as readable lines."""
        if not isinstance(properties, dict) or not properties:
            return ""

        lines = ["Properties:"]
        for name, value in properties.items():
            if isinstance(value, (list, tuple)):
                rendered = ", ".join(str(item) for item in value)
            else:
                rendered = GroundedAnswerTemplate._annotate_weekday(str(value))
            lines.append(f"  {name}: {rendered}")
        return "\n".join(lines)

    @staticmethod
    def _annotate_weekday(value: str) -> str:
        """Append the weekday to an ISO calendar date so the model never guesses it.

        The weekday is computed from the date itself; a value that is not a plain
        ``YYYY-MM-DD`` date is returned unchanged.
        """
        text = value.strip()
        try:
            day = date.fromisoformat(text)
        except ValueError:
            return value
        if len(text) != 10:
            return value
        return f"{value} ({day.strftime('%A')})"


# Registry of available templates, keyed by the name used in configuration.
_TEMPLATES: dict[str, type[PromptTemplate]] = {
    "grounded": GroundedAnswerTemplate,
}


def build_prompt_template(name: str = "grounded") -> PromptTemplate:
    """Resolve a prompt template by name.

    Args:
        name: Template identifier (e.g. ``"grounded"``).

    Returns:
        A ready-to-use :class:`PromptTemplate` instance.

    Raises:
        ValueError: When ``name`` is not a registered template.
    """
    key = name.strip().lower()
    try:
        return _TEMPLATES[key]()
    except KeyError as exc:
        available = ", ".join(sorted(_TEMPLATES))
        raise ValueError(
            f"Unknown prompt template '{name}'. Available: {available}."
        ) from exc
