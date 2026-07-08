
from fastapi import APIRouter

from app.services.qdrant import get_qdrant_client
from app.services.ollama import get_ollama_client

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    qdrant_client = get_qdrant_client()
    ollama_client = get_ollama_client()
    return {
        "status": "ok",
        "qdrant": "connected",
        "ollama": [m.model for m in ollama_client.list().models]
    }