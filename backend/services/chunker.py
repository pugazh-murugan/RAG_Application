import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.schemas import Chunk, RawDoc

_SEPARATORS = ["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", ", ", " ", ""]


def make_doc_id(source_type: str, key: str) -> str:
    return hashlib.sha1(f"{source_type}:{key}".encode("utf-8")).hexdigest()[:16]


def doc_id_for(doc: RawDoc) -> str:
    return make_doc_id(doc.source_type, doc.url or doc.source_name)


def chunk_documents(docs: list[RawDoc]) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
    )
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for doc in docs:
        doc_id = doc_id_for(doc)
        for piece in splitter.split_text(doc.text):
            piece = piece.strip()
            if not piece:
                continue
            idx = counters.get(doc_id, 0)
            counters[doc_id] = idx + 1
            chunks.append(
                Chunk(
                    text=piece,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-{idx}",
                    source_type=doc.source_type,
                    source_name=doc.source_name,
                    url=doc.url,
                    page=doc.page,
                )
            )
    return chunks
