import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from backend.config import get_settings
from backend.schemas import UploadResponse
from backend.services import parser
from backend.services.ingest import index_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


def _safe_name(name: str) -> str:
    base = Path(name).name
    return re.sub(r"[^\w.\- ()஀-௿]", "_", base) or "upload"


def _process(data: bytes, filename: str) -> int:
    docs = parser.parse_file(data, filename)
    if not docs:
        raise ValueError("No readable text found in the file")
    return index_documents(docs)


@router.post("/file", response_model=UploadResponse)
async def upload_file(file: UploadFile) -> UploadResponse:
    s = get_settings()
    filename = _safe_name(file.filename or "upload")
    if Path(filename).suffix.lower() not in parser.SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(parser.SUPPORTED_EXTENSIONS))}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB limit")

    s.upload_dir.mkdir(parents=True, exist_ok=True)
    (s.upload_dir / filename).write_bytes(data)
    try:
        chunks = await asyncio.to_thread(_process, data, filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload pipeline failed for %s", filename)
        raise HTTPException(500, f"Failed to index {filename}: {exc}") from exc
    return UploadResponse(status="indexed", source=filename, chunks=chunks, source_type="upload")
