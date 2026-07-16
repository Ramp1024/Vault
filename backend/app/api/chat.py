from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.chat_service import get_chat_service

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


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

    return StreamingResponse(
        chat_service.stream_answer(message, context=context),
        media_type="text/plain; charset=utf-8",
    )
