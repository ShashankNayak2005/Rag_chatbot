"""
tests/test_retriever.py
Unit tests for retriever module: TF-IDF vector fitting, cosine similarity, threshold cutoff, and expansion.
"""

import pytest
from src.text_chunker import PassageChunk
from src.retriever import TFIDFRetriever, normalize_text


@pytest.fixture
def sample_chunks():
    return [
        PassageChunk(
            chunk_id=1,
            source_filename="doc1.txt",
            page_number=1,
            text="Python is an interpreted, high-level, general-purpose programming language.",
            char_start=0,
            char_end=74,
        ),
        PassageChunk(
            chunk_id=2,
            source_filename="doc1.txt",
            page_number=1,
            text="FastAPI is a modern, fast web framework for building APIs with Python based on standard type hints.",
            char_start=75,
            char_end=176,
        ),
        PassageChunk(
            chunk_id=3,
            source_filename="doc1.txt",
            page_number=2,
            text="Streamlit enables data scientists to build interactive dashboard applications rapidly.",
            char_start=0,
            char_end=86,
        ),
    ]


def test_retriever_empty_chunks():
    with pytest.raises(ValueError, match="empty chunks list"):
        TFIDFRetriever([])


def test_retriever_ranked_retrieval(sample_chunks):
    retriever = TFIDFRetriever(sample_chunks)
    results = retriever.retrieve("How do APIs work with FastAPI?", top_k=2)

    assert len(results) > 0
    top = results[0]
    assert top.chunk.chunk_id == 2
    assert "FastAPI" in top.chunk.text
    assert top.similarity_score > 0.2


def test_retriever_threshold_filtering(sample_chunks):
    retriever = TFIDFRetriever(sample_chunks)
    # High threshold should filter out low similarity matches
    results = retriever.retrieve("Python web APIs", threshold=0.95)
    assert len(results) == 0


def test_retriever_blank_or_zero_vocab_query(sample_chunks):
    retriever = TFIDFRetriever(sample_chunks)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("xyznonexistentword12345") == []


def test_normalize_text():
    raw = "  Hello,   WORLD!\n\tThis is a Test.  "
    norm = normalize_text(raw)
    assert norm == "hello, world! this is a test."
