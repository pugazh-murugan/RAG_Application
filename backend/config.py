from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"

    embedding_provider: str = "huggingface"
    embedding_model: str = "intfloat/multilingual-e5-base"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    uzhavan_base_url: str = "https://uzhavan.tn.gov.in"
    scrape_min_pages: int = 10
    scrape_max_pages: int = 12
    scrape_delay_min: float = 1.0
    scrape_delay_max: float = 2.0
    scrape_retries: int = 3
    scrape_timeout_ms: int = 30000

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    max_upload_mb: int = 25

    data_dir: Path = ROOT / "data"
    upload_dir: Path = ROOT / "data" / "uploads"
    excel_path: Path = ROOT / "data" / "uzhavan_pages.xlsx"
    vector_dir: Path = ROOT / "vector_store"


@lru_cache
def get_settings() -> Settings:
    return Settings()
