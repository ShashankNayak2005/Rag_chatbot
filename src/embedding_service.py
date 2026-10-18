"""
embedding_service.py
Local embedding generation using Sentence Transformers ('sentence-transformers/all-MiniLM-L6-v2').
Operates completely locally on device with zero external API calls.
"""

from typing import List, Optional, Dict, Any
import os

# Module-level model cache to prevent reloading large weights into memory on each instance
_MODEL_CACHE: Dict[str, Any] = {}
# Query embedding cache to avoid recomputing identical queries during chat/eval
_EMBEDDING_CACHE: Dict[str, List[float]] = {}
_MAX_EMBEDDING_CACHE_SIZE = 4096


class EmbeddingServiceError(Exception):
    """Base exception for local embedding generation failures."""
    pass


class EmbeddingService:
    """Local text embedding service using Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = _MODEL_CACHE.get(model_name)

    def _get_model(self):
        """Lazy loader for SentenceTransformer model with global in-memory caching."""
        if self._model is not None:
            return self._model

        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingServiceError(
                "The 'sentence-transformers' library is required for local embeddings. "
                "Install it using: pip install sentence-transformers"
            ) from e

        try:
            # First attempt loading from local cache with local_files_only=True to eliminate 30+ second network lag
            try:
                model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                # Fallback to downloading if not cached locally yet
                model = SentenceTransformer(self.model_name)

            try:
                import torch
                # Enable inference optimizations
                if hasattr(torch, "set_num_threads") and torch.get_num_threads() < 4:
                    torch.set_num_threads(4)
            except Exception:
                pass
            _MODEL_CACHE[self.model_name] = model
            self._model = model
            return self._model
        except Exception as e:
            raise EmbeddingServiceError(
                f"Failed to initialize embedding model '{self.model_name}': {str(e)}. "
                "Note: an active internet connection is required for the initial model download from Hugging Face."
            ) from e

    @property
    def max_seq_length(self) -> int:
        """Return the maximum sequence length (in tokens) supported by the underlying model."""
        model = self._get_model()
        return getattr(model, "max_seq_length", 256)

    def check_truncation(self, text: str) -> bool:
        """Check if a given text would exceed the tokenizer limit and be truncated."""
        cleaned = text.strip()
        if not cleaned:
            return False
        model = self._get_model()
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            tokens = model.tokenizer.tokenize(cleaned)
            return len(tokens) > self.max_seq_length
        # Fallback heuristic: 1 token is ~3.5 characters on average
        return len(cleaned) > (self.max_seq_length * 3.5)

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized embedding vector for a single string with memory caching."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot generate embedding for empty text.")

        cache_key = f"{self.model_name}:{cleaned}"
        if cache_key in _EMBEDDING_CACHE:
            return _EMBEDDING_CACHE[cache_key]

        model = self._get_model()
        if self.check_truncation(cleaned):
            import logging
            logging.getLogger(__name__).warning(
                f"Text exceeds model max_seq_length ({self.max_seq_length} tokens). "
                "Tokens beyond this limit will be truncated. Use appropriately sized chunks."
            )

        try:
            try:
                import torch
                with torch.inference_mode():
                    embedding = model.encode(
                        cleaned,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            except Exception:
                embedding = model.encode(
                    cleaned,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

            result = embedding.tolist()
            if len(_EMBEDDING_CACHE) < _MAX_EMBEDDING_CACHE_SIZE:
                _EMBEDDING_CACHE[cache_key] = result
            return result
        except Exception as e:
            raise EmbeddingServiceError(f"Embedding inference failed: {str(e)}") from e

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Batch generate normalized embedding vectors for multiple strings using torch inference mode."""
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts]
        model = self._get_model()
        try:
            try:
                import torch
                with torch.inference_mode():
                    embeddings = model.encode(
                        cleaned_texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            except Exception:
                embeddings = model.encode(
                    cleaned_texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise EmbeddingServiceError(f"Batch embedding inference failed: {str(e)}") from e

