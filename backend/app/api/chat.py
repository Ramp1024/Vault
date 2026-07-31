from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.answer import Citation
from app.models.context import AssembledContext
from app.services.answer_service import get_answer_service

router = APIRouter()

# Max characters of chunk text to include as a preview snippet in each source.
_SNIPPET_LIMIT = 200

# Control byte separating the streamed answer text from the trailing JSON
# metadata frame. A NUL never appears in model text or JSON, so it unambiguously
# marks the end of the answer body without constraining what the model may emit.
_FINAL_FRAME_DELIMITER = "\x00"


class ChatRequest(BaseModel):
    message: str


def _serialize_sources(context: AssembledContext) -> list[dict]:
    """Shape the assembled context into a compact JSON payload for the UI.

    Each source carries the stable ``reference_id`` the answer cites (``[1]``),
    so the frontend can align validated citations with the sources shown.
    """
    serialized = []
    for item in context.chunks:
        chunk = item.chunk
        snippet = " ".join(chunk.content.split())[:_SNIPPET_LIMIT]
        serialized.append(
            {
                "reference_id": item.reference_id,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "title": chunk.document_title,
                "score": round(item.score, 6),
                "snippet": snippet,
            }
        )
    return serialized


def _serialize_citations(citations: tuple[Citation, ...]) -> list[dict]:
    """Shape validated citations for the final metadata frame."""
    return [
        {
            "reference_id": citation.reference_id,
            "document_id": citation.document_id,
            "chunk_id": citation.chunk_id,
            "title": citation.document_title,
        }
        for citation in citations
    ]


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    answer_service = get_answer_service()

    try:
        # Retrieve + build context/prompt before opening the stream so failures
        # can surface as proper HTTP errors.
        context = answer_service.prepare(message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to prepare chat response: {exc}") from exc

    def stream() -> Iterator[str]:
        # First frame: a single JSON line describing the retrieved sources,
        # terminated by a newline. Everything after the first newline is the
        # streamed answer text.
        yield json.dumps({"sources": _serialize_sources(context.context)}) + "\n"

        # Stream the raw answer text unchanged while accumulating it so the
        # backend can validate citations against the retrieved context once
        # generation completes (never incrementally during streaming).
        parts: list[str] = []
        start = time.perf_counter()
        for token in answer_service.stream_answer(message, context=context):
            parts.append(token)
            yield token
        latency_ms = (time.perf_counter() - start) * 1000.0

        citations = answer_service.map_citations("".join(parts), context.context)
        final_frame = {
            "type": "final",
            "citations": _serialize_citations(citations),
            "latency": {"generation_ms": round(latency_ms, 1)},
        }
        # Final frame: delimiter byte followed by the JSON metadata. The backend
        # is the single source of truth for citations; the UI must render these
        # rather than the raw inline markers in the answer text.
        yield _FINAL_FRAME_DELIMITER + json.dumps(final_frame)

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
    )

