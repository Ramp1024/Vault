from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.core.config import settings
from app.models.prompt import Prompt
from app.services.ollama import get_ollama_client


class LLM(ABC):
    """Backend-independent large language model interface.

    Generation depends only on this abstraction, never on a concrete backend.
    New providers (OpenAI, Anthropic, ...) implement this interface without any
    changes to the answer generator or the rest of the pipeline.
    """

    @abstractmethod
    def generate(self, prompt: Prompt) -> str:
        """Synchronously generate a completion for ``prompt`` and return its text."""
        raise NotImplementedError

    def stream(self, prompt: Prompt) -> Iterator[str]:
        """Stream generated text fragments as they arrive.

        Optional capability; backends that do not support streaming raise
        ``NotImplementedError``. Reserved for a future streaming milestone.
        """
        raise NotImplementedError("streaming generation is not supported by this backend")


class OllamaLLM(LLM):
    """LLM backed by a locally running Ollama server."""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
        # Optional structured-output constraint (e.g. "json" or a JSON schema).
        # When set, Ollama constrains generation to valid output of that shape,
        # which both guarantees parseable results and keeps generation short.
        self.response_format = response_format
        # Optional fixed decoding seed. With temperature 0 this makes generation
        # reproducible run-to-run; None leaves seeding to the backend default.
        self.seed = seed
        self.client = get_ollama_client()

    def _options(self) -> dict[str, float | int]:
        # Low temperature keeps answers grounded in the supplied context; a
        # negative/zero max_tokens is treated as "no explicit cap".
        options: dict[str, float | int] = {"temperature": self.temperature}
        if self.max_tokens and self.max_tokens > 0:
            options["num_predict"] = self.max_tokens
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def _format_kwargs(self) -> dict:
        return {"format": self.response_format} if self.response_format else {}

    def generate(self, prompt: Prompt) -> str:
        response = self.client.generate(
            model=self.model,
            prompt=prompt.user,
            system=prompt.system,
            options=self._options(),
            **self._format_kwargs(),
        )
        return str(response["response"])

    def stream(self, prompt: Prompt) -> Iterator[str]:
        stream = self.client.generate(
            model=self.model,
            prompt=prompt.user,
            system=prompt.system,
            options=self._options(),
            stream=True,
            **self._format_kwargs(),
        )
        for chunk in stream:
            text = str(chunk.get("response", ""))
            if text:
                yield text


class MockLLM(LLM):
    """Deterministic, dependency-free LLM for tests and offline evaluation.

    Returns a canned response (optionally cycling through a supplied list) so the
    generation pipeline can be exercised end-to-end without a running backend.
    """

    def __init__(self, responses: list[str] | str | None = None) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self._responses = list(responses) if responses else []
        self._calls = 0

    @property
    def calls(self) -> int:
        """Number of times ``generate`` has been invoked."""
        return self._calls

    def generate(self, prompt: Prompt) -> str:
        index = self._calls
        self._calls += 1
        if self._responses:
            return self._responses[index % len(self._responses)]
        # Default: echo a grounded-looking answer that cites the first source if
        # the prompt exposes one, so citation mapping has something to resolve.
        citation = "[1]" if "[1]" in prompt.user else ""
        return f"This is a mock answer. {citation}".strip()

    def stream(self, prompt: Prompt) -> Iterator[str]:
        yield self.generate(prompt)


# Registry of available backends, keyed by the name used in configuration.
_BACKENDS = {"ollama", "mock"}


def build_llm(backend: str | None = None) -> LLM:
    """Construct the configured LLM backend.

    Args:
        backend: Backend name override; falls back to ``settings.LLM_BACKEND``.

    Returns:
        A ready-to-use :class:`LLM` instance.

    Raises:
        ValueError: When the backend name is not recognized.
    """
    name = (backend or settings.LLM_BACKEND).strip().lower()
    if name == "ollama":
        return OllamaLLM()
    if name == "mock":
        return MockLLM()
    available = ", ".join(sorted(_BACKENDS))
    raise ValueError(f"Unknown LLM backend '{backend}'. Available: {available}.")


def build_intent_llm(backend: str | None = None) -> LLM:
    """Construct an LLM tuned for schema-aware intent analysis.

    Intent analysis needs a short, strictly-structured JSON response rather than
    free-form prose. For the Ollama backend this constrains output to JSON,
    forces deterministic decoding (temperature 0), and caps the token budget, so
    generation stays fast and bounded — avoiding the long, open-ended completions
    that can exceed the request timeout — while always yielding parseable output.
    """
    name = (backend or settings.LLM_BACKEND).strip().lower()
    if name == "ollama":
        return OllamaLLM(
            temperature=0.0,
            max_tokens=settings.INTENT_LLM_MAX_TOKENS,
            response_format="json",
            seed=settings.INTENT_LLM_SEED,
        )
    if name == "mock":
        return MockLLM()
    available = ", ".join(sorted(_BACKENDS))
    raise ValueError(f"Unknown LLM backend '{backend}'. Available: {available}.")
