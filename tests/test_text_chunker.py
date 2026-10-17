"""
tests/test_text_chunker.py
Unit tests for text_chunker module: sliding window chunking, lookback, and metadata tracking.
"""

import pytest
from src.document_loader import ExtractedDocument, DocumentPage
from src.text_chunker import TextChunker, chunk_document


def test_chunker_initialization_invalid_arguments():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        TextChunker(chunk_size=100, chunk_overlap=-5)
    with pytest.raises(ValueError, match="must be strictly less"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_split_short_text():
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    slices = chunker.split_text("A quick brown fox jumps over the lazy dog.")
    assert len(slices) == 1
    assert slices[0][0] == 0
    assert slices[0][2] == "A quick brown fox jumps over the lazy dog."


def test_split_long_text_with_overlap():
    text = (
        "Artificial Intelligence represents a major technological leap. "
        "Retrieval Augmented Generation combines retrieval with generation. "
        "Vector databases store dense embedding representations. "
        "TF-IDF provides a robust, transparent baseline without external dependencies."
    )
    chunker = TextChunker(chunk_size=90, chunk_overlap=25)
    slices = chunker.split_text(text)
    assert len(slices) >= 3

    for start, end, chunk_str in slices:
        assert len(chunk_str) > 0
        assert chunk_str in text


def test_chunk_document_preserves_page_numbers():
    doc = ExtractedDocument(
        filename="solar_report.pdf",
        file_type="pdf",
        pages=[
            DocumentPage(page_number=1, text="Solar energy systems harvest photovoltaic power from sunlight."),
            DocumentPage(page_number=2, text="Inverters convert DC electricity into AC grid power."),
        ],
        total_characters=115,
    )

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 2

    assert chunks[0].chunk_id == 1
    assert chunks[0].page_number == 1
    assert chunks[0].source_filename == "solar_report.pdf"
    assert "Solar energy" in chunks[0].text

    assert chunks[1].chunk_id == 2
    assert chunks[1].page_number == 2
    assert chunks[1].source_filename == "solar_report.pdf"
    assert "Inverters convert" in chunks[1].text
