
from fastapi import APIRouter

from app.services.qdrant import get_qdrant_client
from app.services.qdrant_service import QdrantService
from app.services.ollama import get_ollama_client

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    qdrant_client = get_qdrant_client()
    qdrant_service = QdrantService(qdrant_client)
    ollama_client = get_ollama_client()
    qdrant_status = "connected" if qdrant_service.health_check() else "disconnected"

    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "ollama": [m.model for m in ollama_client.list().models]
    }