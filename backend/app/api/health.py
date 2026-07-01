
from fastapi import APIRouter

from services.chroma import get_chroma_client
from services.ollama import get_ollama_client

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    chroma = get_chroma_client()
    ollama_client = get_ollama_client()
    return {
        "status": "ok",
        "chroma": chroma.heartbeat(),
        "ollama": [m.model for m in ollama_client.list().models]
    }