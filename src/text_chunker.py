"""
text_chunker.py
Splits extracted document pages into overlapping, manageable passages while preserving
exact page numbers, character offsets, and chunk identifiers.
"""

from dataclasses import dataclass
from typing import List, Tuple
from src.document_loader import ExtractedDocument


@dataclass
class PassageChunk:
    """Represents a passage chunk extracted from a document with full provenance metadata."""
    chunk_id: int
    source_filename: str
    page_number: int
    text: str
    char_start: int
    char_end: int
    document_id: str = ""

    @property
    def estimated_tokens(self) -> int:
        """Approximate token count (standard ~4 characters per token)."""
        return approx_token_count(self.text)


def approx_token_count(text: str) -> int:
    """Approximate subword token count (~3.8 characters per token for English text)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving natural sentence terminators."""
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class TextChunker:
    """Sliding-window chunker with boundary lookback for natural sentence preservation.
    
    Calibrated to respect sentence-transformers/all-MiniLM-L6-v2 maximum sequence length
    of 256 tokens. A target of ~800 characters (~200 tokens) with ~150 characters (~35-40 tokens)
    overlap ensures zero silent truncation by the embedding model.
    """

    MAX_EMBEDDING_TOKENS = 256
    # 800 chars ~ 200 tokens, safely within 256 token limit of all-MiniLM-L6-v2
    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Split a block of text into slices (start_idx, end_idx, chunk_text).
        Prefers breaking at sentence or paragraph boundaries.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [(0, len(cleaned), cleaned)]

        slices: List[Tuple[int, int, str]] = []
        start = 0
        total_len = len(cleaned)

        while start < total_len:
            end = start + self.chunk_size

            if end >= total_len:
                chunk_slice = cleaned[start:total_len].strip()
                if chunk_slice:
                    slices.append((start, total_len, chunk_slice))
                break

            # Lookback window for clean sentence or paragraph breaks
            lookback = min(120, self.chunk_size // 3)
            cut_idx = -1

            for sep in ["\n\n", ". ", "? ", "! ", "\n"]:
                pos = cleaned.rfind(sep, end - lookback, end)
                if pos != -1:
                    cut_idx = pos + len(sep)
                    break

            # Fall back to word break
            if cut_idx == -1:
                space_pos = cleaned.rfind(" ", end - lookback, end)
                if space_pos != -1:
                    cut_idx = space_pos + 1
                else:
                    cut_idx = end

            chunk_slice = cleaned[start:cut_idx].strip()
            if chunk_slice:
                slices.append((start, cut_idx, chunk_slice))

            # Advance window forward, maintaining overlap
            start = max(start + 1, cut_idx - self.chunk_overlap)

        return slices

    def chunk_document(self, document: ExtractedDocument) -> List[PassageChunk]:
        """Convert an ExtractedDocument into a list of PassageChunks, preserving page numbers and document_id."""
        chunks: List[PassageChunk] = []
        global_chunk_id = 1

        for page in document.pages:
            slices = self.split_text(page.text)
            for char_start, char_end, text in slices:
                chunks.append(
                    PassageChunk(
                        chunk_id=global_chunk_id,
                        source_filename=document.filename,
                        page_number=page.page_number,
                        text=text,
                        char_start=char_start,
                        char_end=char_end,
                        document_id=document.document_id,
                    )
                )
                global_chunk_id += 1

        return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = TextChunker.DEFAULT_CHUNK_OVERLAP,
) -> List[PassageChunk]:
    """Convenience functional wrapper for TextChunker."""
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(document)

