from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["upload", "web"]


@dataclass
class RawDoc:
    text: str
    source_type: str
    source_name: str
    url: str = ""
    page: int = 0


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    source_type: str
    source_name: str
    url: str
    page: int


class UploadResponse(BaseModel):
    status: str
    source: str
    chunks: int
    source_type: SourceType


class ScrapeResponse(BaseModel):
    status: str
    pages_scraped: int
    excel: str
    chunks: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    stream: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class DocumentInfo(BaseModel):
    id: str
    source_name: str
    source_type: SourceType
    url: str
    chunks: int


class DeleteResponse(BaseModel):
    status: str
    id: str
    chunks_removed: int
