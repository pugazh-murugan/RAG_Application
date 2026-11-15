import asyncio
import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from backend.config import get_settings
from backend.schemas import ChatResponse
from backend.services import embedder, vector_store
from backend.services.vector_store import Hit

logger = logging.getLogger(__name__)

NOT_FOUND = "I could not find this in the indexed data."

_SYSTEM_PROMPT = f"""You are the Uzhavan assistant for Tamil Nadu farmers and agriculture officers.
Rules:
1. Answer ONLY using the CONTEXT provided. Never use outside knowledge.
2. If the context does not contain the answer, reply with exactly: {NOT_FOUND}
3. Reply in the same language as the question (Tamil question -> Tamil answer, English question -> English answer).
4. Keep the answer simple, accurate and concise. Do not invent figures, dates or scheme names."""

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        s = get_settings()
        if not s.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not configured in .env")
        _client = AsyncOpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url or None)
    return _client


async def retrieve(question: str) -> list[Hit]:
    query_vec = await asyncio.to_thread(embedder.embed_query, question)
    return await asyncio.to_thread(vector_store.search, query_vec, get_settings().top_k)


def _messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(f"[{i}] Source: {h.label}\n{h.text}" for i, h in enumerate(hits, start=1))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]


_SOURCE_SCORE_MARGIN = 0.05


def _sources(hits: list[Hit]) -> list[str]:
    """List only sources whose similarity is close to the best match, not every retrieved chunk."""
    best = min(h.distance for h in hits)
    seen: list[str] = []
    for h in hits:
        if h.distance - best <= _SOURCE_SCORE_MARGIN and h.label not in seen:
            seen.append(h.label)
    return seen


def _is_not_found(answer: str) -> bool:
    return answer.strip().startswith(NOT_FOUND)


async def answer_question(question: str) -> ChatResponse:
    hits = await retrieve(question)
    if not hits:
        return ChatResponse(answer=NOT_FOUND, sources=[])
    resp = await _get_client().chat.completions.create(
        model=get_settings().llm_model, messages=_messages(question, hits), temperature=0
    )
    answer = (resp.choices[0].message.content or "").strip() or NOT_FOUND
    return ChatResponse(answer=answer, sources=[] if _is_not_found(answer) else _sources(hits))


async def stream_answer(question: str) -> AsyncIterator[dict]:
    """Yields {"type": "token", "text": ...} events and a final {"type": "sources", "sources": [...]}."""
    hits = await retrieve(question)
    if not hits:
        yield {"type": "token", "text": NOT_FOUND}
        yield {"type": "sources", "sources": []}
        return
    stream = await _get_client().chat.completions.create(
        model=get_settings().llm_model, messages=_messages(question, hits), temperature=0, stream=True
    )
    answer = ""
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if delta:
            answer += delta
            yield {"type": "token", "text": delta}
    yield {"type": "sources", "sources": [] if _is_not_found(answer) else _sources(hits)}
