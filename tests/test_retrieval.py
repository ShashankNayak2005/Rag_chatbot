"""
test_retrieval.py
Unit and integration tests for semantic retrieval, scoring, filtering,
and multi-document isolation in ChromaDB.
"""

from pathlib import Path
import pytest
from src.document_loader import load_document
from src.text_chunker import chunk_document
from src.embedding_service import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever, TFIDFRetriever
from src.config import settings


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingService(settings.embedding_model)


@pytest.fixture
def temp_retriever(tmp_path, embedder):
    db_path = str(tmp_path / "chroma_retrieval_test")
    vstore = ChromaVectorStore(persist_directory=db_path, collection_name="test_retrieval")
    return SemanticRetriever(vstore, embedder, distance_threshold=0.85, top_k=3)


def test_retrieval_returns_relevant_chunks(temp_retriever):
    doc_text = (
        "Solar photovoltaic panels convert sunlight directly into electricity using the photoelectric effect. "
        "Silicon semiconductors absorb photons and release free electrons. "
        "Wind turbines generate kinetic energy from air currents."
    )
    doc = load_document(doc_text.encode("utf-8"), filename="energy.txt")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)
    temp_retriever.index_document(doc, chunks)

    results = temp_retriever.retrieve(
        document_id=doc.document_id,
        query="How do solar photovoltaic panels generate power?",
        top_k=2,
    )
    assert len(results) > 0
    assert any("photovoltaic" in r.text.lower() for r in results)
    assert results[0].distance < 0.85
    assert results[0].source_filename == "energy.txt"


def test_document_isolation_in_retrieval(temp_retriever):
    """Ensure retrieving from doc A never returns chunks from doc B."""
    doc_a = load_document(b"AlphaCorp was established in Paris in 2010.", filename="alpha.txt")
    chunks_a = chunk_document(doc_a, chunk_size=200, chunk_overlap=20)
    temp_retriever.index_document(doc_a, chunks_a)

    doc_b = load_document(b"BetaCorp was established in Tokyo in 2022.", filename="beta.txt")
    chunks_b = chunk_document(doc_b, chunk_size=200, chunk_overlap=20)
    temp_retriever.index_document(doc_b, chunks_b)

    # Query doc A for BetaCorp content -> must NOT find BetaCorp chunks
    results = temp_retriever.retrieve(
        document_id=doc_a.document_id,
        query="Where was BetaCorp established?",
        top_k=2,
    )
    for r in results:
        assert r.source_filename == "alpha.txt"
        assert "BetaCorp" not in r.text


def test_tfidf_keyword_retrieval():
    doc_text = "Datavex headquarters is located in Bengaluru. Main services: AI solutions."
    doc = load_document(doc_text.encode("utf-8"), filename="datavex.txt")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)

    tfidf = TFIDFRetriever(chunks)
    ranked = tfidf.retrieve("bengaluru headquarters", top_k=1)
    assert len(ranked) == 1
    assert "bengaluru" in ranked[0].chunk.text.lower()
