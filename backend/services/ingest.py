import logging

from backend.schemas import RawDoc
from backend.services import chunker, embedder, vector_store

logger = logging.getLogger(__name__)


def index_documents(docs: list[RawDoc], replace_existing: bool = True) -> int:
    """Shared tail of both pipelines: chunk -> embed -> store."""
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        return 0
    if replace_existing:
        for doc_id in {c.doc_id for c in chunks}:
            vector_store.delete_by_doc_id(doc_id)
    embeddings = embedder.embed_passages([c.text for c in chunks])
    stored = vector_store.add_chunks(chunks, embeddings)
    logger.info("Indexed %d chunks", stored)
    return stored
