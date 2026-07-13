import os

from app.services.ollama import get_ollama_client


class GenerationService:
    """Thin wrapper around Ollama text generation."""

    MODEL = os.environ.get("OLLAMA_GENERATE_MODEL", "llama3.1:8b")

    def __init__(self):
        self.client = get_ollama_client()

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return raw generated text."""
        response = self.client.generate(model=self.MODEL, prompt=prompt)
        return str(response["response"])


def get_generation_service() -> GenerationService:
    """Factory function to create a generation service instance."""
    return GenerationService()
