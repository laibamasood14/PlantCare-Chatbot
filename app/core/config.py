from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    
    TEMPERATURE: float = 0.3  # keep this, it's reused

    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings — multilingual model for cross-lingual support
    # AFTER (~170MB vs ~350MB, still covers Urdu+English well)
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"

    # Paths
    PDF_DIR: str = "pdfs"
    VECTORSTORE_DIR: str = "vectorstore"

    # Chunking
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # Retrieval
    TOP_K_DENSE: int = 6
    TOP_K_BM25: int = 6
    TOP_K_RERANK: int = 4

    # App
    APP_TITLE: str = "Plant Care RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()

