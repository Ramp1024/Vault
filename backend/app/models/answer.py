from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    """A resolved citation pointing back at a retrieved chunk.

    Produced by the citation mapper, never by the LLM: the model only emits
    reference ids (``[1]``) and the mapper resolves each id to the concrete
    chunk/document metadata so citations always refer to retrieved content.
    """

    reference_id: int
    chunk_id: str
    document_id: str
    document_title: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """Strongly typed generation result returned instead of raw text.

    ``confidence`` is intentionally optional and defaults to ``None`` so future
    milestones can populate it (or add further fields) without breaking callers.
    """

    answer: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    confidence: float | None = None
