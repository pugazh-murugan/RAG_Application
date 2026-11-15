import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers import chat, documents, scrape, upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Uzhavan RAG Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_s = get_settings()
_s.upload_dir.mkdir(parents=True, exist_ok=True)
_s.vector_dir.mkdir(parents=True, exist_ok=True)

app.include_router(upload.router)
app.include_router(scrape.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
