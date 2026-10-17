from dataclasses import dataclass
from typing import List
import re

from backend.app.services.extractor import ExtractionResult


@dataclass
class DocumentChunk:
    chunk_id: int
    source_filename: str
    page_number: int
    text: str
    char_start: int
    char_end: int


def split_text_into_chunks(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[tuple[int, int, str]]:
    """Split a single text string into sliding chunks with overlap, preferring whitespace boundaries."""
    if not text:
        return []

    # If text is shorter than chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [(0, len(text), text)]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunk_slice = text[start:text_length].strip()
            if chunk_slice:
                chunks.append((start, text_length, chunk_slice))
            break

        # Look for sentence boundary (. ! ? \n) near the end to avoid cutting mid-sentence
        lookback_window = min(80, chunk_size // 3)
        candidate_cut = -1
        for sep in [". ", "? ", "! ", "\n\n", "\n"]:
            pos = text.rfind(sep, end - lookback_window, end)
            if pos != -1:
                candidate_cut = pos + len(sep)
                break

        # If no sentence boundary found, try space
        if candidate_cut == -1:
            space_pos = text.rfind(" ", end - lookback_window, end)
            if space_pos != -1:
                candidate_cut = space_pos + 1
            else:
                candidate_cut = end

        chunk_slice = text[start:candidate_cut].strip()
        if chunk_slice:
            chunks.append((start, candidate_cut, chunk_slice))

        # Advance start with overlap
        start = max(start + 1, candidate_cut - chunk_overlap)

    return chunks


def create_chunks_from_document(
    extraction: ExtractionResult,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[DocumentChunk]:
    """Create a list of DocumentChunks from an ExtractionResult, preserving page numbers."""
    all_chunks: List[DocumentChunk] = []
    global_chunk_id = 1

    for page in extraction.pages:
        slices = split_text_into_chunks(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for char_start, char_end, chunk_text in slices:
            all_chunks.append(
                DocumentChunk(
                    chunk_id=global_chunk_id,
                    source_filename=extraction.filename,
                    page_number=page.page_number,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            global_chunk_id += 1

    return all_chunks
