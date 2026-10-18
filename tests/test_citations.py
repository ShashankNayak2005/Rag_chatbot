"""
test_citations.py
Unit tests for the source citation and evidence tracking system.
Verifies PDF page tracking, TXT chunk tracking, and provenance integrity.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.document_loader import load_document
from src.text_chunker import chunk_document
from src.embedding_service import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever
from src.qa_engine import QAEngine, ANSWER_NOT_FOUND_IN_DOCUMENT, strip_source_citation_text
from src.config import settings


def make_mock_completion(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp.choices = [mock_choice]
    return mock_resp


def test_txt_citations_contain_filename_and_chunk_id(tmp_path):
    text = "Datavex was founded in 2020. Headquarters is in Bengaluru. Total employees: 50."
    doc = load_document(text.encode("utf-8"), filename="profile.txt")
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=20)

    embedder = EmbeddingService(settings.embedding_model)
    vstore = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_cite_txt"), collection_name="cite_txt")
    retriever = SemanticRetriever(vstore, embedder, distance_threshold=0.85)
    retriever.index_document(doc, chunks)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_mock_completion("Founded in 2020.")

    qa = QAEngine(retriever=retriever, api_key="dummy", openai_client=mock_client)
    res = qa.answer_question("When was the company founded?", document_id=doc.document_id)

    assert res.is_found is True
    assert res.source_filename == "profile.txt"
    assert res.chunk_id is not None
    assert len(res.citations) > 0
    cite = res.citations[0]
    assert cite["source_filename"] == "profile.txt"
    assert "chunk_id" in cite


def test_pdf_citations_preserve_page_numbers(tmp_path):
    pdf_path = Path("sample_docs/sample_article.pdf")
    if not pdf_path.exists():
        pytest.skip("sample_article.pdf not found")

    doc = load_document(pdf_path)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)

    embedder = EmbeddingService(settings.embedding_model)
    vstore = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_cite_pdf"), collection_name="cite_pdf")
    retriever = SemanticRetriever(vstore, embedder, distance_threshold=0.85)
    retriever.index_document(doc, chunks)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_mock_completion("The report discusses artificial intelligence.")

    qa = QAEngine(retriever=retriever, api_key="dummy", openai_client=mock_client)
    res = qa.answer_question("What does the report discuss?", document_id=doc.document_id)

    assert res.is_found is True
    assert res.source_filename == "sample_article.pdf"
    assert res.page_number is not None
    assert res.page_number >= 1


def test_no_citations_for_unsupported_questions(tmp_path):
    text = "Datavex was founded in 2020. Headquarters is in Bengaluru."
    doc = load_document(text.encode("utf-8"), filename="test.txt")
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=20)

    embedder = EmbeddingService(settings.embedding_model)
    vstore = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_cite_none"), collection_name="cite_none")
    retriever = SemanticRetriever(vstore, embedder, distance_threshold=0.85)
    retriever.index_document(doc, chunks)

    mock_client = MagicMock()
    qa = QAEngine(retriever=retriever, api_key="dummy", openai_client=mock_client)
    res = qa.answer_question("What is the annual revenue?", document_id=doc.document_id)

    assert res.answer == ANSWER_NOT_FOUND_IN_DOCUMENT
    assert res.supporting_passage is None
    assert res.source_filename is None


def test_strip_source_citation_text_variations():
    """Verify that redundant trailing citation strings are cleanly stripped from answer text."""
    # Pattern: Source: file, page, chunk
    raw1 = "The company was founded in **2020**.\n\nSource: test.pdf, page 1, chunk 1"
    assert strip_source_citation_text(raw1) == "The company was founded in **2020**."

    # Pattern: **Source:** file, page, chunk
    raw2 = "Founded in 2020.\n\n**Source:** data.pdf, page 2, chunk 3"
    assert strip_source_citation_text(raw2) == "Founded in 2020."

    # Pattern: Bracketed
    raw3 = "Revenue reached $5M. [test.txt, page 1, chunk 2]"
    assert strip_source_citation_text(raw3) == "Revenue reached $5M."

    # Pattern: Parenthesized
    raw4 = "Solar efficiency is 24%. (Source: report.pdf, page 3, chunk 1)"
    assert strip_source_citation_text(raw4) == "Solar efficiency is 24%."

    # Pattern: Negative refusal remains intact
    assert strip_source_citation_text(ANSWER_NOT_FOUND_IN_DOCUMENT) == ANSWER_NOT_FOUND_IN_DOCUMENT

    # Pattern: Normal text without trailing citation
    clean = "Datavex was founded in 2020 in Bengaluru."
    assert strip_source_citation_text(clean) == clean


def test_qa_engine_strips_duplicate_citation_from_llm_answer(tmp_path):
    """Verify QAEngine removes LLM-appended 'Source: ...' citations from res.answer while keeping metadata."""
    text = "Datavex was founded in 2020. Headquarters is in Bengaluru."
    doc = load_document(text.encode("utf-8"), filename="test.txt")
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=20)

    embedder = EmbeddingService(settings.embedding_model)
    vstore = ChromaVectorStore(persist_directory=str(tmp_path / "chroma_cite_strip"), collection_name="cite_strip")
    retriever = SemanticRetriever(vstore, embedder, distance_threshold=0.85)
    retriever.index_document(doc, chunks)

    mock_client = MagicMock()
    # LLM returns text with appended Source line
    mock_client.chat.completions.create.return_value = make_mock_completion(
        "The company was founded in **2020**.\n\nSource: test.pdf, page 1, chunk 1"
    )

    qa = QAEngine(retriever=retriever, api_key="dummy", openai_client=mock_client)
    res = qa.answer_question("When was the company founded?", document_id=doc.document_id)

    assert res.is_found is True
    # The duplicate text citation must be stripped from res.answer
    assert res.answer == "The company was founded in **2020**."
    assert "Source:" not in res.answer
    # But the metadata fields must be preserved
    assert res.source_filename == "test.txt"
    assert res.page_number is not None
    assert res.chunk_id is not None

