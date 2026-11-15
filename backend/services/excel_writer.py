import logging
from pathlib import Path

import pandas as pd

from backend.schemas import RawDoc
from backend.services.scraper import ScrapedPage

logger = logging.getLogger(__name__)

COLUMNS = ["S.No", "URL", "Title", "Headings", "Content", "Scraped_At"]
_CELL_LIMIT = 32000


def write_excel(pages: list[ScrapedPage], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "S.No": i,
            "URL": p.url,
            "Title": p.title,
            "Headings": " | ".join(p.headings)[:_CELL_LIMIT],
            "Content": p.content[:_CELL_LIMIT],
            "Scraped_At": p.scraped_at,
        }
        for i, p in enumerate(pages, start=1)
    ]
    pd.DataFrame(rows, columns=COLUMNS).to_excel(path, index=False, engine="openpyxl")
    logger.info("Wrote %d rows to %s", len(rows), path)
    return len(rows)


def read_excel_as_docs(path: Path) -> list[RawDoc]:
    df = pd.read_excel(path, dtype=str, engine="openpyxl").fillna("")
    docs: list[RawDoc] = []
    for _, row in df.iterrows():
        title = row["Title"].strip()
        parts = [title, row["Headings"].replace(" | ", "\n"), row["Content"]]
        text = "\n".join(p for p in parts if p.strip()).strip()
        if not text:
            continue
        docs.append(
            RawDoc(
                text=text,
                source_type="web",
                source_name=title or row["URL"],
                url=row["URL"].strip(),
                page=0,
            )
        )
    return docs
