"""Application configuration module.

Handles environment variables and system settings for the RAG pipeline.
"""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


@dataclass
class Settings:
    """Configuration settings for embeddings, ChromaDB, and Groq API."""

    # Groq LLM API Configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.groq.com/openai/v1")
    groq_chat_model: str = os.getenv("GROQ_MODEL") or os.getenv("GROQ_CHAT_MODEL") or os.getenv("DEEPSEEK_CHAT_MODEL", "qwen/qwen3.8-27b")

    # Local Embeddings & Vector Store
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    retrieval_distance_threshold: float = float(os.getenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.85"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    retrieval_top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    @property
    def has_api_key(self) -> bool:
        """Check whether a valid API key has been configured."""
        key = self.groq_api_key.strip()
        return bool(key) and key not in ("your_api_key_here", "your_groq_api_key_here")

    @property
    def active_api_key(self) -> str:
        return self.groq_api_key

    @property
    def active_base_url(self) -> str:
        return self.groq_base_url

    @property
    def active_model(self) -> str:
        return self.groq_chat_model


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


settings = get_settings()
