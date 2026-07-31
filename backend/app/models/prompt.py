from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """A backend-independent prompt split into system and user messages.

    Keeping the two parts separate lets each LLM backend map them to its native
    shape (Ollama's ``system``/``prompt`` fields, OpenAI chat messages, ...)
    without the prompt builder knowing which backend will run.
    """

    system: str
    user: str

    def to_text(self) -> str:
        """Flatten to a single string for completion-style backends."""
        return f"{self.system}\n\n{self.user}"
