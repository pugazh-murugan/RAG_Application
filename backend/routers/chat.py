import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.schemas import ChatRequest, ChatResponse
from backend.services import rag

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


async def _ndjson(question: str) -> AsyncIterator[str]:
    try:
        async for event in rag.stream_answer(question):
            yield json.dumps(event, ensure_ascii=False) + "\n"
    except Exception as exc:
        logger.exception("Streaming chat failed")
        yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"


@router.post("/chat", response_model=None)
async def chat(req: ChatRequest) -> ChatResponse | StreamingResponse:
    question = req.question.strip()
    if req.stream:
        return StreamingResponse(_ndjson(question), media_type="application/x-ndjson")
    try:
        return await rag.answer_question(question)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(500, f"Chat failed: {exc}") from exc
