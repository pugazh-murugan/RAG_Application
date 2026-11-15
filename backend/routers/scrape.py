import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import ROOT, get_settings
from backend.schemas import ScrapeResponse
from backend.services import excel_writer, scraper, vector_store
from backend.services.ingest import index_documents

logger = logging.getLogger(__name__)
router = APIRouter(tags=["scrape"])

_scrape_lock = asyncio.Lock()


def _index_from_excel() -> int:
    s = get_settings()
    docs = excel_writer.read_excel_as_docs(s.excel_path)
    vector_store.delete_by_source_type("web")
    return index_documents(docs, replace_existing=False)


@router.post("/scrape/uzhavan", response_model=ScrapeResponse)
async def scrape_uzhavan() -> ScrapeResponse:
    if _scrape_lock.locked():
        raise HTTPException(409, "A scrape is already running")
    async with _scrape_lock:
        s = get_settings()
        try:
            pages = await scraper.scrape_uzhavan()
        except Exception as exc:
            logger.exception("Scraping failed")
            raise HTTPException(502, f"Scraping failed: {exc}") from exc
        if not pages:
            raise HTTPException(502, "No pages could be scraped from the Uzhavan website")
        try:
            await asyncio.to_thread(excel_writer.write_excel, pages, s.excel_path)
            chunks = await asyncio.to_thread(_index_from_excel)
        except Exception as exc:
            logger.exception("Web pipeline failed after scraping")
            raise HTTPException(500, f"Indexing failed: {exc}") from exc
        return ScrapeResponse(
            status="completed",
            pages_scraped=len(pages),
            excel=s.excel_path.relative_to(ROOT).as_posix(),
            chunks=chunks,
        )


@router.get("/download/excel")
async def download_excel() -> FileResponse:
    path = get_settings().excel_path
    if not path.exists():
        raise HTTPException(404, "No Excel file yet. Run the scraper first.")
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
