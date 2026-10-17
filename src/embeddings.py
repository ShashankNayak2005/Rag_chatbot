"""
embeddings.py
Local Sentence Transformers embedding generation module alias.
"""

from src.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingServiceError",
]
