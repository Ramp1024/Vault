from app.core.config import settings
from app.services.ollama import get_ollama_client


class GenerationService:
    """Thin wrapper around Ollama text generation."""

    def __init__(self):
        self.model = settings.GENERATION_MODEL
        self.client = get_ollama_client()

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return raw generated text."""
        response = self.client.generate(model=self.model, prompt=prompt)
        return str(response["response"])


def get_generation_service() -> GenerationService:
    """Factory function to create a generation service instance."""
    return GenerationService()
