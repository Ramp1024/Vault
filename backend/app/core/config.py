import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    NOTION_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_EMBED_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_TIMEOUT_SECONDS: float = float(
        os.environ.get("OLLAMA_TIMEOUT_SECONDS", "90")
    )
    QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "Vault")
    QDRANT_UPSERT_BATCH_SIZE: int = int(
        os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "100")
    )

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")


settings = Settings()
