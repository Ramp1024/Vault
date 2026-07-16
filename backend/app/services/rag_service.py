"""Compatibility shim for older imports.

RAG orchestration now lives in `app.services.chat_service.ChatService`.
"""

from app.services.chat_service import ChatService, get_chat_service

RAGService = ChatService


def get_rag_service() -> ChatService:
    return get_chat_service()
