"""
tests/test_acceptance.py
Mandatory Acceptance Tests (TC1 - TC7) and edge-case validation suite.
"""

import io
from pathlib import Path
import pymupdf as fitz
import pytest

from src.document_loader import (
    load_document,
    DocumentValidationError,
    EmptyFileError,
    UnsupportedFileTypeError,
    UnreadableFileError,
    NoExtractableTextError,
)
from src.text_chunker import chunk_document, TextChunker
from src.retriever import TFIDFRetriever
from src.qa_engine import QAEngine, ANSWER_NOT_FOUND_IN_DOCUMENT


@pytest.fixture
def datavex_qa_engine() -> QAEngine:
    """Fixture providing QAEngine indexed with the official Datavex test document."""
    doc_path = Path("data/datavex_profile.txt")
    doc = load_document(doc_path)
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=50)
    retriever = TFIDFRetriever(chunks)
    return QAEngine(retriever, min_similarity_threshold=0.15)


# -----------------------------------------------------------------------------
# MANDATORY ACCEPTANCE TESTS (TC1 - TC7)
# -----------------------------------------------------------------------------

def test_tc1_when_was_company_founded(datavex_qa_engine: QAEngine):
    """TC1: 'When was the company founded?' -> Answer contains 2020 and cites supporting passage."""
    res = datavex_qa_engine.answer_question("When was the company founded?")
    assert res.is_found is True
    assert "2020" in res.answer
    assert res.supporting_passage is not None
    assert "Founded: 2020" in res.supporting_passage
    assert res.source_filename == "datavex_profile.txt"
    assert res.similarity_score > 0.15


def test_tc2_where_is_headquarters(datavex_qa_engine: QAEngine):
    """TC2: 'Where is the headquarters?' -> Answer contains Bengaluru and cites supporting passage."""
    res = datavex_qa_engine.answer_question("Where is the headquarters?")
    assert res.is_found is True
    assert "bengaluru" in res.answer.lower()
    assert res.supporting_passage is not None
    assert "Headquarters: Bengaluru" in res.supporting_passage
    assert res.source_filename == "datavex_profile.txt"
    assert res.similarity_score > 0.15


def test_tc3_how_many_employees(datavex_qa_engine: QAEngine):
    """TC3: 'How many employees are there?' -> Answer contains 50 and cites supporting passage."""
    res = datavex_qa_engine.answer_question("How many employees are there?")
    assert res.is_found is True
    assert "50" in res.answer
    assert res.supporting_passage is not None
    assert "Employees: 50" in res.supporting_passage
    assert res.source_filename == "datavex_profile.txt"
    assert res.similarity_score > 0.15


def test_tc4_who_is_ceo_not_found(datavex_qa_engine: QAEngine):
    """TC4: 'Who is the CEO?' -> Exact answer: ANSWER_NOT_FOUND_IN_DOCUMENT."""
    res = datavex_qa_engine.answer_question("Who is the CEO?")
    assert res.is_found is False
    assert res.answer == ANSWER_NOT_FOUND_IN_DOCUMENT
    assert res.supporting_passage is None
    assert res.source_filename is None


def test_tc5_empty_document_validation():
    """TC5: Empty document -> Clear validation error; do not crash."""
    with pytest.raises(EmptyFileError) as exc_info:
        load_document(b"", filename="empty.txt")
    assert "is empty" in str(exc_info.value)

    # Empty PDF stream
    with pytest.raises(EmptyFileError) as exc_info_pdf:
        load_document(b"", filename="empty.pdf")
    assert "is empty" in str(exc_info_pdf.value)


def test_tc6_annual_revenue_not_found(datavex_qa_engine: QAEngine):
    """TC6: 'What is the company's annual revenue?' -> Exact answer: ANSWER_NOT_FOUND_IN_DOCUMENT."""
    res = datavex_qa_engine.answer_question("What is the company's annual revenue?")
    assert res.is_found is False
    assert res.answer == ANSWER_NOT_FOUND_IN_DOCUMENT
    assert res.supporting_passage is None
    assert res.source_filename is None


def test_tc7_in_which_city_is_company_based(datavex_qa_engine: QAEngine):
    """TC7: 'In which city is the company based?' -> Answer contains Bengaluru."""
    res = datavex_qa_engine.answer_question("In which city is the company based?")
    assert res.is_found is True
    assert "bengaluru" in res.answer.lower()
    assert res.supporting_passage is not None
    assert "Headquarters: Bengaluru" in res.supporting_passage


# -----------------------------------------------------------------------------
# ADDITIONAL EDGE-CASE AND INTEGRITY TESTS
# -----------------------------------------------------------------------------

def test_unsupported_file_extension():
    """Rejects unsupported extensions like .docx or .csv."""
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        load_document(b"dummy data", filename="profile.docx")
    assert "Unsupported file type" in str(exc_info.value)


def test_corrupted_pdf_handling():
    """Corrupted PDF bytes raise UnreadableFileError with clear message."""
    bad_pdf = b"%PDF-1.4\ncorrupt header and damaged binary stream\x00\xff"
    with pytest.raises(UnreadableFileError) as exc_info:
        load_document(bad_pdf, filename="corrupt.pdf")
    assert "Failed to open/parse PDF" in str(exc_info.value)


def test_scanned_image_only_pdf_handling():
    """PDF with no extractable text raises NoExtractableTextError."""
    doc = fitz.open()
    doc.new_page(width=500, height=700)  # Blank page without text
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    with pytest.raises(NoExtractableTextError) as exc_info:
        load_document(buf.getvalue(), filename="scanned.pdf")
    assert "contains no extractable text" in str(exc_info.value)


def test_blank_question_validation(datavex_qa_engine: QAEngine):
    """Blank or whitespace-only questions raise ValueError."""
    with pytest.raises(ValueError, match="cannot be blank"):
        datavex_qa_engine.answer_question("   ")


def test_multi_page_pdf_provenance():
    """Verify that page numbers are preserved across multi-page PDFs."""
    doc = fitz.open()
    p1 = doc.new_page(width=500, height=700)
    p1.insert_text((50, 100), "Chapter 1: The Foundation of Quantum Computing was laid in 1980.")
    p2 = doc.new_page(width=500, height=700)
    p2.insert_text((50, 100), "Chapter 2: Quantum supremacy was demonstrated in 2019 using Sycamore processor.")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    extracted = load_document(buf.getvalue(), filename="quantum.pdf")
    assert len(extracted.pages) == 2
    assert extracted.pages[0].page_number == 1
    assert extracted.pages[1].page_number == 2

    chunks = chunk_document(extracted, chunk_size=300, chunk_overlap=30)
    retriever = TFIDFRetriever(chunks)
    qa = QAEngine(retriever)

    # Query page 2 topic
    res_p2 = qa.answer_question("What processor demonstrated quantum supremacy in 2019?")
    assert res_p2.is_found is True
    assert "Sycamore" in res_p2.answer
    assert res_p2.page_number == 2
    assert res_p2.source_filename == "quantum.pdf"


def test_false_positive_guardrail(datavex_qa_engine: QAEngine):
    """Out-of-domain and completely fabricated questions must return ANSWER_NOT_FOUND_IN_DOCUMENT."""
    fabricated_questions = [
        "What is the capital of Australia?",
        "Who is the lead guitarist of Led Zeppelin?",
        "How many satellites are in geostationary orbit?",
        "What are the ingredients of chocolate chip cookies?",
    ]
    for q in fabricated_questions:
        res = datavex_qa_engine.answer_question(q)
        assert res.answer == ANSWER_NOT_FOUND_IN_DOCUMENT, f"Failed false-positive guard for: {q}"
        assert res.supporting_passage is None
