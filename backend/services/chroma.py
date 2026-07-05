import chromadb
from backend.app.core.config import CHROMA_HOST, CHROMA_PORT


def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
