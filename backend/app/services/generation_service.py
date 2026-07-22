from app.core.config import settings
from app.services.ollama import get_ollama_client


class GenerationService:
    """Thin wrapper around Ollama text generation."""

    def __init__(self):
        self.model = settings.GENERATION_MODEL
        self.client = get_ollama_client()
        # Low temperature keeps answers grounded in the retrieved context
        # instead of drifting into the model's own training knowledge.
        self.options = {"temperature": settings.GENERATION_TEMPERATURE}

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return raw generated text."""
        response = self.client.generate(
            model=self.model, prompt=prompt, options=self.options
        )
        return str(response["response"])

    def stream_generate(self, prompt: str):
        """Stream generated text chunks from Ollama as they arrive."""
        stream = self.client.generate(
            model=self.model, prompt=prompt, stream=True, options=self.options
        )
        for chunk in stream:
            text = str(chunk.get("response", ""))
            if text:
                yield text


def get_generation_service() -> GenerationService:
    """Factory function to create a generation service instance."""
    return GenerationService()
