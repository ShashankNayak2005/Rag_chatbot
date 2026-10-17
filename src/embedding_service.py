"""
embedding_service.py
Local embedding generation using Sentence Transformers ('sentence-transformers/all-MiniLM-L6-v2').
Operates completely locally on device with zero external API calls.
"""

from typing import List, Optional
import os


class EmbeddingServiceError(Exception):
    """Base exception for local embedding generation failures."""
    pass


class EmbeddingService:
    """Local text embedding service using Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingServiceError(
                "The 'sentence-transformers' library is required for local embeddings. "
                "Install it using: pip install sentence-transformers"
            ) from e

        try:
            # Load local model (initial load downloads from Hugging Face if not cached)
            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as e:
            raise EmbeddingServiceError(
                f"Failed to initialize embedding model '{self.model_name}': {str(e)}. "
                "Note: an active internet connection is required for the initial model download from Hugging Face."
            ) from e

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized embedding vector for a single string."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot generate embedding for empty text.")

        model = self._get_model()
        try:
            # normalize_embeddings=True ensures cosine distance can be computed accurately
            embedding = model.encode(cleaned, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingServiceError(f"Embedding inference failed: {str(e)}") from e

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch generate normalized embedding vectors for multiple strings."""
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts]
        model = self._get_model()
        try:
            embeddings = model.encode(
                cleaned_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise EmbeddingServiceError(f"Batch embedding inference failed: {str(e)}") from e
