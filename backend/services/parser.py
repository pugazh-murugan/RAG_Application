import io
import logging
import re
from pathlib import Path

import pandas as pd
from docx import Document
from pypdf import PdfReader

from backend.schemas import RawDoc

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _frame_to_text(df: pd.DataFrame) -> str:
    df = df.dropna(how="all").fillna("")
    lines = []
    for _, row in df.iterrows():
        cells = [f"{col}: {str(val).strip()}" for col, val in row.items() if str(val).strip()]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _parse_pdf(data: bytes, name: str) -> list[RawDoc]:
    reader = PdfReader(io.BytesIO(data))
    return [
        RawDoc(text=page.extract_text() or "", source_type="upload", source_name=name, page=i)
        for i, page in enumerate(reader.pages, start=1)
    ]


def _parse_docx(data: bytes, name: str) -> list[RawDoc]:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [RawDoc(text="\n".join(parts), source_type="upload", source_name=name, page=1)]


def _parse_txt(data: bytes, name: str) -> list[RawDoc]:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return [RawDoc(text=data.decode(encoding), source_type="upload", source_name=name, page=1)]
        except UnicodeDecodeError:
            continue
    return [RawDoc(text=data.decode("utf-8", errors="ignore"), source_type="upload", source_name=name, page=1)]


def _parse_csv(data: bytes, name: str) -> list[RawDoc]:
    df = pd.read_csv(io.BytesIO(data), dtype=str, encoding_errors="ignore")
    return [RawDoc(text=_frame_to_text(df), source_type="upload", source_name=name, page=1)]


def _parse_xlsx(data: bytes, name: str) -> list[RawDoc]:
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
    return [
        RawDoc(text=f"Sheet: {sheet}\n{_frame_to_text(df)}", source_type="upload", source_name=name, page=i)
        for i, (sheet, df) in enumerate(sheets.items(), start=1)
    ]


_PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".txt": _parse_txt,
    ".csv": _parse_csv,
    ".xlsx": _parse_xlsx,
}


def parse_file(data: bytes, filename: str) -> list[RawDoc]:
    ext = Path(filename).suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")
    docs = parser(data, filename)
    cleaned = []
    for doc in docs:
        doc.text = clean_text(doc.text)
        if doc.text:
            cleaned.append(doc)
    logger.info("Parsed %s into %d non-empty sections", filename, len(cleaned))
    return cleaned
