import ollama
from app.core.config import settings


def get_ollama_client() -> ollama.Client:
    return ollama.Client(
        host=settings.OLLAMA_BASE_URL,
        timeout=settings.OLLAMA_TIMEOUT_SECONDS,
    )


def assert_ollama_reachable() -> None:
    """Fail fast when Ollama is not reachable at startup."""
    try:
        get_ollama_client().list()
    except Exception as exc:
        raise RuntimeError(
            "Ollama startup check failed. Ensure Ollama is installed in WSL, "
            "start it with 'ollama serve', pull the required models, and "
            "set OLLAMA_BASE_URL if you are not using the default local endpoint. "
            f"Current OLLAMA_BASE_URL={settings.OLLAMA_BASE_URL}. "
            f"Original error: {exc}"
        ) from exc