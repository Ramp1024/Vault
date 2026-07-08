import ollama
from app.core.config import settings

def get_ollama_client() -> ollama.Client:
    return ollama.Client(host=settings.OLLAMA_BASE_URL)