import ollama
from core.config import OLLAMA_BASE_URL

def get_ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_BASE_URL)