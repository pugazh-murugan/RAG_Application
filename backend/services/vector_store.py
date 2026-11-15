import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from backend.config import get_settings
from backend.schemas import Chunk

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_index: faiss.IndexIDMap2 | None = None
_meta: dict[int, dict] = {}
_loaded = False


@dataclass
class Hit:
    text: str
    source_type: str
    source_name: str
    url: str
    page: int
    distance: float

    @property
    def label(self) -> str:
        if self.source_type == "web" and self.url:
            return self.url
        return f"{self.source_name} (page {self.page})" if self.page else self.source_name


def _paths() -> tuple[Path, Path]:
    d = get_settings().vector_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / "index.faiss", d / "metadata.json"


def _int_id(chunk_id: str) -> int:
    return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


def _ensure_loaded() -> None:
    global _index, _meta, _loaded
    if _loaded:
        return
    index_file, meta_file = _paths()
    if index_file.exists() and meta_file.exists():
        _index = faiss.read_index(str(index_file))
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        _meta = {int(k): v for k, v in raw.items()}
        logger.info("Loaded FAISS index with %d vectors", _index.ntotal)
    _loaded = True


def _persist() -> None:
    index_file, meta_file = _paths()
    if _index is None:
        return
    tmp_index = index_file.with_suffix(".faiss.tmp")
    tmp_meta = meta_file.with_suffix(".json.tmp")
    faiss.write_index(_index, str(tmp_index))
    tmp_meta.write_text(json.dumps({str(k): v for k, v in _meta.items()}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_index, index_file)
    os.replace(tmp_meta, meta_file)


def _as_matrix(vectors: list[list[float]]) -> np.ndarray:
    mat = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(mat)
    return mat


def _remove(ids: list[int]) -> int:
    if not ids or _index is None:
        return 0
    _index.remove_ids(np.asarray(ids, dtype="int64"))
    for i in ids:
        _meta.pop(i, None)
    _persist()
    return len(ids)


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    global _index
    if not chunks:
        return 0
    mat = _as_matrix(embeddings)
    ids = [_int_id(c.chunk_id) for c in chunks]
    with _lock:
        _ensure_loaded()
        if _index is None:
            _index = faiss.IndexIDMap2(faiss.IndexFlatIP(mat.shape[1]))
        elif _index.d != mat.shape[1]:
            raise RuntimeError(
                f"Embedding size {mat.shape[1]} does not match the index ({_index.d}). "
                "Delete the vector_store files and re-index after changing the embedding model."
            )
        _index.remove_ids(np.asarray(ids, dtype="int64"))
        _index.add_with_ids(mat, np.asarray(ids, dtype="int64"))
        for i, c in zip(ids, chunks):
            _meta[i] = {
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "source_type": c.source_type,
                "source_name": c.source_name,
                "url": c.url,
                "page": c.page,
            }
        _persist()
    return len(chunks)


def delete_by_doc_id(doc_id: str) -> int:
    with _lock:
        _ensure_loaded()
        return _remove([i for i, m in _meta.items() if m["doc_id"] == doc_id])


def delete_by_source_type(source_type: str) -> int:
    with _lock:
        _ensure_loaded()
        return _remove([i for i, m in _meta.items() if m["source_type"] == source_type])


def search(query_embedding: list[float], k: int) -> list[Hit]:
    with _lock:
        _ensure_loaded()
        if _index is None or _index.ntotal == 0:
            return []
        scores, ids = _index.search(_as_matrix([query_embedding]), min(k, _index.ntotal))
        hits = []
        for score, i in zip(scores[0], ids[0]):
            meta = _meta.get(int(i))
            if i == -1 or meta is None:
                continue
            hits.append(
                Hit(
                    text=meta["text"],
                    source_type=meta["source_type"],
                    source_name=meta["source_name"],
                    url=meta.get("url", ""),
                    page=int(meta.get("page", 0)),
                    distance=float(1.0 - score),
                )
            )
        return hits


def list_documents() -> list[dict]:
    with _lock:
        _ensure_loaded()
        docs: dict[str, dict] = {}
        for meta in _meta.values():
            entry = docs.setdefault(
                meta["doc_id"],
                {
                    "id": meta["doc_id"],
                    "source_name": meta["source_name"],
                    "source_type": meta["source_type"],
                    "url": meta.get("url", ""),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: (d["source_type"], d["source_name"]))
