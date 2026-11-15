import logging
import threading

from backend.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_local_model = None
_openai_client = None


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _get_local_model():
    global _local_model
    with _lock:
        if _local_model is None:
            from sentence_transformers import SentenceTransformer

            name = get_settings().embedding_model
            logger.info("Loading embedding model %s", name)
            _local_model = SentenceTransformer(name)
    return _local_model


def _get_openai_client():
    global _openai_client
    with _lock:
        if _openai_client is None:
            from openai import OpenAI

            s = get_settings()
            key = s.openai_api_key or s.llm_api_key
            if not key:
                raise RuntimeError("OPENAI_API_KEY (or LLM_API_KEY) is required for OpenAI embeddings")
            _openai_client = OpenAI(api_key=key)
    return _openai_client


def _embed(texts: list[str], prefix: str) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    if s.embedding_provider == "openai":
        client = _get_openai_client()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            resp = client.embeddings.create(model=s.openai_embedding_model, input=batch)
            vectors.extend(d.embedding for d in resp.data)
        return vectors
    model = _get_local_model()
    if _is_e5(s.embedding_model):
        texts = [f"{prefix}{t}" for t in texts]
    return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "passage: ")


def embed_query(text: str) -> list[float]:
    return _embed([text], "query: ")[0]
