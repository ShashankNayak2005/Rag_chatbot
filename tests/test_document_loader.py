"""
tests/test_document_loader.py
Unit tests for document_loader module: TXT, PDF, empty files, and error handling.
"""

import io
from pathlib import Path
import pymupdf as fitz
import pytest

from src.document_loader import (
    load_document,
    validate_file_metadata,
    UnsupportedFileTypeError,
    EmptyFileError,
    UnreadableFileError,
    NoExtractableTextError,
)


def test_validate_metadata_valid():
    assert validate_file_metadata("test.txt", b"content") == ".txt"
    assert validate_file_metadata("TEST.PDF", b"%PDF...") == ".pdf"


def test_validate_metadata_empty_filename():
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_metadata("", b"content")


def test_validate_metadata_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        validate_file_metadata("report.docx", b"some bytes")


def test_validate_metadata_empty_bytes():
    with pytest.raises(EmptyFileError, match="is empty"):
        validate_file_metadata("notes.txt", b"")


def test_load_valid_txt(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Company: Datavex\nFounded: 2020", encoding="utf-8")

    doc = load_document(txt_file)
    assert doc.filename == "sample.txt"
    assert doc.file_type == "txt"
    assert len(doc.pages) == 1
    assert "Datavex" in doc.pages[0].text
    assert doc.total_characters == len("Company: Datavex\nFounded: 2020")


def test_load_txt_from_bytes():
    raw_bytes = b"Hello world, this is a plain text file."
    doc = load_document(raw_bytes, filename="hello.txt")
    assert doc.file_type == "txt"
    assert doc.pages[0].text == "Hello world, this is a plain text file."


def test_load_whitespace_only_txt():
    with pytest.raises(NoExtractableTextError, match="no readable text"):
        load_document(b"   \n\n\t   ", filename="whitespace.txt")


def test_load_valid_pdf():
    pdf = fitz.open()
    p1 = pdf.new_page()
    p1.insert_text((50, 100), "Page 1 Content: Cloud Architecture.")
    p2 = pdf.new_page()
    p2.insert_text((50, 100), "Page 2 Content: Scalable Microservices.")
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()

    doc = load_document(buf.getvalue(), filename="arch.pdf")
    assert doc.file_type == "pdf"
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Page 1 Content" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Page 2 Content" in doc.pages[1].text


def test_load_scanned_image_only_pdf():
    pdf = fitz.open()
    pdf.new_page()  # Blank page without text characters
    buf = io.BytesIO()
    pdf.save(buf)
    pdf.close()

    with pytest.raises(NoExtractableTextError, match="no extractable text"):
        load_document(buf.getvalue(), filename="scanned.pdf")


def test_load_corrupt_pdf():
    bad_pdf = b"%PDF-1.5 corrupt data"
    with pytest.raises(UnreadableFileError, match="Failed to open/parse PDF"):
        load_document(bad_pdf, filename="bad.pdf")
