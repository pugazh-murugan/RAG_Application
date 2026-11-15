import asyncio

from fastapi import APIRouter, HTTPException

from backend.schemas import DeleteResponse, DocumentInfo
from backend.services import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in await asyncio.to_thread(vector_store.list_documents)]


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str) -> DeleteResponse:
    removed = await asyncio.to_thread(vector_store.delete_by_doc_id, doc_id)
    if removed == 0:
        raise HTTPException(404, "Document not found")
    return DeleteResponse(status="deleted", id=doc_id, chunks_removed=removed)
