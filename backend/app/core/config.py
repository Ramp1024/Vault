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
    # Path to the filterable metadata schema discovered during ingestion. The
    # schema is the contract consumed by the schema-aware LLM intent analyzer.
    METADATA_SCHEMA_PATH: str = os.environ.get(
        "METADATA_SCHEMA_PATH",
        str(Path(__file__).resolve().parents[2] / "config" / "metadata_schema.json"),
    )
    # Optional user configuration for role-based property classification. The
    # file (JSON) may define ``property_roles`` (per-property overrides) and
    # ``type_roles`` (connector native-type -> role overrides). It is optional:
    # when absent, connectors fall back to their built-in default role mapping.
    PROPERTY_ROLES_PATH: str = os.environ.get(
        "PROPERTY_ROLES_PATH",
        str(Path(__file__).resolve().parents[2] / "config" / "property_roles.json"),
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

    # Schema-aware LLM intent analysis. When enabled the chat pipeline wraps the
    # deterministic rule-based analyzer in a CompositeQueryAnalyzer that also
    # consults an LLM to rewrite conversational queries and infer metadata
    # filters from the discovered MetadataSchema. Disabled by default so behavior
    # is unchanged out of the box; the LLM only acts as a query compiler and the
    # retrieval engine is untouched.
    INTENT_ANALYZER_ENABLED: bool = os.environ.get(
        "INTENT_ANALYZER_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    # Token cap for the intent analyzer's structured JSON response. The output is
    # a small search request, so a tight cap keeps generation fast and bounded.
    INTENT_LLM_MAX_TOKENS: int = int(os.environ.get("INTENT_LLM_MAX_TOKENS", "512"))
    # Fixed decoding seed for the intent analyzer. Combined with temperature 0
    # this makes filter inference reproducible run-to-run, so the same query
    # resolves to the same date filter instead of drifting between requests.
    INTENT_LLM_SEED: int = int(os.environ.get("INTENT_LLM_SEED", "0"))
    # Acceptance thresholds for LLM-proposed constraints (augmenting analyzer).
    # A candidate constraint is only allowed to influence retrieval when its
    # grounding score AND model confidence both clear these bars — the LLM is a
    # proposal engine, never the decision-maker. Conservative by design: raising
    # them makes the analyzer accept fewer LLM constraints (safer, narrower).
    INTENT_LLM_MIN_CONFIDENCE: float = float(
        os.environ.get("INTENT_LLM_MIN_CONFIDENCE", "0.6")
    )
    INTENT_GROUNDING_THRESHOLD: float = float(
        os.environ.get("INTENT_GROUNDING_THRESHOLD", "0.6")
    )
    # IANA timezone (e.g. "Asia/Kolkata", "America/New_York") used to anchor
    # relative-date resolution ("yesterday", "day before yesterday"). Containers
    # default to UTC, which shifts relative dates by a day for users ahead of or
    # behind UTC; set this to your local zone so "today" matches your wall clock.
    APP_TIMEZONE: str = os.environ.get("APP_TIMEZONE", "UTC")
    # The record timestamp field that authoring-activity questions ("what did I
    # write/do <when>") filter on. The LLM emits a temporal descriptor, resolved
    # in code to a range over this top-level field (defaults to last-edited time).
    AUTHORSHIP_DATE_FIELD: str = os.environ.get(
        "AUTHORSHIP_DATE_FIELD", "last_edited_time"
    )

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")


settings = Settings()
