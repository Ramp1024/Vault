import os

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHROMA_HOST: str = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT: str = os.environ.get("CHROMA_PORT", "8001")