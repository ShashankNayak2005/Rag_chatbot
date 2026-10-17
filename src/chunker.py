"""
chunker.py
Sentence-aware chunking module alias and re-exports.
"""

from src.text_chunker import (
    PassageChunk,
    chunk_document,
    approx_token_count,
    split_sentences,
)

__all__ = [
    "PassageChunk",
    "chunk_document",
    "approx_token_count",
    "split_sentences",
]
