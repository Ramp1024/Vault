from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.search_result import SearchResult
from app.services.chat_service import get_chat_service

router = APIRouter()

# Max characters of chunk text to include as a preview snippet in each source.
_SNIPPET_LIMIT = 200


class ChatRequest(BaseModel):
    message: str


def _serialize_sources(sources: list[SearchResult]) -> list[dict]:
    """Shape retrieved results into a compact JSON payload for the UI."""
    serialized = []
    for result in sources:
        snippet = " ".join(result.chunk.content.split())[:_SNIPPET_LIMIT]
        serialized.append(
            {
                "chunk_id": result.chunk.id,
                "document_id": result.chunk.document_id,
                "title": result.chunk.document_title,
                "score": round(result.score, 6),
                "snippet": snippet,
            }
        )
    return serialized


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    chat_service = get_chat_service()

    try:
        # Trigger retrieval + prompt build before opening the stream so failures
        # can surface as proper HTTP errors.
        context = chat_service.prepare(message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to prepare chat response: {exc}") from exc

    def stream() -> Iterator[str]:
        # First frame: a single JSON line describing the retrieved sources,
        # terminated by a newline. Everything after the first newline is the
        # streamed answer text. JSON escapes any newlines in the payload, so the
        # first '\n' unambiguously ends the header.
        yield json.dumps({"sources": _serialize_sources(context.sources)}) + "\n"
        yield from chat_service.stream_answer(message, context=context)

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
    )

