import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    NOTION_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    GENERATION_MODEL: str = os.environ.get("GENERATION_MODEL", "llama3.1:8b")
    GENERATION_TEMPERATURE: float = float(
        os.environ.get("GENERATION_TEMPERATURE", "0.2")
    )
    OLLAMA_TIMEOUT_SECONDS: float = float(
        os.environ.get("OLLAMA_TIMEOUT_SECONDS", "90")
    )
    QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "Vault")
    QDRANT_UPSERT_BATCH_SIZE: int = int(
        os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "100")
    )
    BM25_INDEX_PATH: str = os.environ.get(
        "BM25_INDEX_PATH",
        str(Path(__file__).resolve().parents[2] / "config" / "bm25_index.pkl"),
    )
    # Retrieval pipeline configuration. RETRIEVAL_MODE selects which strategies
    # the chat pipeline runs: "vector", "bm25", or "hybrid" (Vector + BM25 fused
    # with Reciprocal Rank Fusion). RRF_K is the RRF rank-damping constant.
    RETRIEVAL_MODE: str = os.environ.get("RETRIEVAL_MODE", "hybrid")
    RRF_K: int = int(os.environ.get("RRF_K", "60"))
    # Cross-encoder reranking. When RERANK_ENABLED is true the chat pipeline
    # retrieves RERANK_CANDIDATE_POOL candidates, reranks them with RERANK_MODEL
    # (a Sentence-Transformers cross-encoder), and keeps the top results. The
    # model is model-agnostic configuration only — changing it requires no code.
    RERANK_ENABLED: bool = os.environ.get("RERANK_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    RERANK_MODEL: str = os.environ.get(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    RERANK_CANDIDATE_POOL: int = int(os.environ.get("RERANK_CANDIDATE_POOL", "25"))
    RERANK_TOP_N: int = int(os.environ.get("RERANK_TOP_N", "5"))
    # Answer generation layer. The LLM is reached only through a backend-agnostic
    # abstraction: LLM_BACKEND selects the implementation ("ollama" or "mock"),
    # and the model, temperature, and token cap tune generation. CONTEXT_TOKEN_BUDGET
    # bounds how much retrieved context the ContextBuilder assembles, and
    # PROMPT_TEMPLATE selects a swappable prompt template. All are configurable via
    # environment variables without code changes. LLM_MODEL/LLM_TEMPERATURE default
    # to the existing generation settings so behavior is unchanged out of the box.
    LLM_BACKEND: str = os.environ.get("LLM_BACKEND", "ollama")
    LLM_MODEL: str = os.environ.get(
        "LLM_MODEL", os.environ.get("GENERATION_MODEL", "llama3.1:8b")
    )
    LLM_TEMPERATURE: float = float(
        os.environ.get(
            "LLM_TEMPERATURE", os.environ.get("GENERATION_TEMPERATURE", "0.2")
        )
    )
    LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "0"))
    CONTEXT_TOKEN_BUDGET: int = int(os.environ.get("CONTEXT_TOKEN_BUDGET", "2000"))
    PROMPT_TEMPLATE: str = os.environ.get("PROMPT_TEMPLATE", "grounded")

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")


settings = Settings()
