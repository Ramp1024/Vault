import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

# OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# CHROMA_HOST: str = os.environ.get("CHROMA_HOST", "127.0.0.1")
# CHROMA_PORT: str = os.environ.get("CHROMA_PORT", "8001")


class Settings(BaseSettings):
    NOTION_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    CHROMA_HOST: str = os.environ.get("CHROMA_HOST", "127.0.0.1")
    CHROMA_PORT: str = os.environ.get("CHROMA_PORT", "8001")

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")


settings = Settings()
