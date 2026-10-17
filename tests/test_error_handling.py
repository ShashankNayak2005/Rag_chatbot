"""
test_error_handling.py
Automated tests for error handling across all layers:
- Document loading validation errors (empty, corrupt, scanned, unsupported)
- Query validation errors (empty, whitespace)
- API errors (auth failure, rate limit, timeout, missing key)
"""

from unittest.mock import MagicMock
import pytest
from openai import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError

from src.document_loader import (
    load_document,
    EmptyFileError,
    UnsupportedFileTypeError,
    UnreadableFileError,
    NoExtractableTextError,
)
from src.qa_engine import QAEngine


def test_empty_file_rejected():
    with pytest.raises(EmptyFileError):
        load_document(b"", filename="empty.txt")


def test_whitespace_only_file_rejected():
    with pytest.raises((EmptyFileError, NoExtractableTextError)):
        load_document(b"   \n\n\t  \r\n", filename="whitespace.txt")


def test_unsupported_file_extension_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        load_document(b"Some text", filename="document.docx")


def test_corrupt_pdf_rejected():
    corrupt_bytes = b"%PDF-1.4\nSome random garbage that is not valid PDF structure\n%%EOF"
    with pytest.raises(UnreadableFileError):
        load_document(corrupt_bytes, filename="corrupt.pdf")


def test_empty_question_raises_value_error():
    qa = QAEngine()
    with pytest.raises(ValueError):
        qa.answer_question("")

    with pytest.raises(ValueError):
        qa.answer_question("   \t\n  ")


def test_missing_api_key_status():
    mock_retriever = MagicMock()
    mock_passage = MagicMock()
    mock_passage.text = "Company founded in 2020."
    mock_passage.similarity_score = 0.95
    mock_passage.distance = 0.05
    mock_passage.source_filename = "doc.txt"
    mock_passage.page_number = 1
    mock_passage.chunk_id = 0
    mock_retriever.retrieve.return_value = [mock_passage]

    qa = QAEngine(retriever=mock_retriever, api_key="")
    res = qa.answer_question("When was the company founded?", document_id="doc-123", use_llm=True)

    assert res.status == "api_error"
    assert "API key" in (res.error_message or "")


def test_api_authentication_error():
    mock_retriever = MagicMock()
    mock_passage = MagicMock()
    mock_passage.text = "Company founded in 2020."
    mock_passage.similarity_score = 0.95
    mock_passage.distance = 0.05
    mock_passage.source_filename = "doc.txt"
    mock_passage.page_number = 1
    mock_passage.chunk_id = 0
    mock_retriever.retrieve.return_value = [mock_passage]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key provided",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid API Key"}},
    )

    qa = QAEngine(retriever=mock_retriever, api_key="invalid_key", openai_client=mock_client)
    res = qa.answer_question("When was the company founded?", document_id="doc-123", use_llm=True)

    assert res.status == "api_error"
    assert "Authentication Failed" in (res.error_message or "")


def test_api_rate_limit_error():
    mock_retriever = MagicMock()
    mock_passage = MagicMock()
    mock_passage.text = "Company founded in 2020."
    mock_passage.similarity_score = 0.95
    mock_passage.distance = 0.05
    mock_passage.source_filename = "doc.txt"
    mock_passage.page_number = 1
    mock_passage.chunk_id = 0
    mock_retriever.retrieve.return_value = [mock_passage]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limit exceeded"}},
    )

    qa = QAEngine(retriever=mock_retriever, api_key="valid_key", openai_client=mock_client)
    res = qa.answer_question("When was the company founded?", document_id="doc-123", use_llm=True)

    assert res.status == "api_error"
    assert "Rate Limit Exceeded" in (res.error_message or "")


def test_api_timeout_error():
    mock_retriever = MagicMock()
    mock_passage = MagicMock()
    mock_passage.text = "Company founded in 2020."
    mock_passage.similarity_score = 0.95
    mock_passage.distance = 0.05
    mock_passage.source_filename = "doc.txt"
    mock_passage.page_number = 1
    mock_passage.chunk_id = 0
    mock_retriever.retrieve.return_value = [mock_passage]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(
        request=MagicMock()
    )

    qa = QAEngine(retriever=mock_retriever, api_key="valid_key", openai_client=mock_client)
    res = qa.answer_question("When was the company founded?", document_id="doc-123", use_llm=True)

    assert res.status == "api_error"
    assert "Timed Out" in (res.error_message or "")
