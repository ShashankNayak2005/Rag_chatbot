import pytest
from backend.app.services.extractor import (
    extract_document,
    extract_text_from_txt,
    extract_text_from_pdf,
    validate_file_metadata,
)
from backend.app.services.exceptions import (
    UnsupportedFileTypeError,
    EmptyFileError,
    UnreadableDocumentError,
    NoExtractableTextError,
)


def test_validate_file_metadata_valid():
    ext = validate_file_metadata("report.PDF", b"some content")
    assert ext == ".pdf"
    ext_txt = validate_file_metadata("notes.txt", b"notes content")
    assert ext_txt == ".txt"


def test_validate_file_metadata_unsupported_type():
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        validate_file_metadata("document.docx", b"some bytes")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        validate_file_metadata("image.png", b"some bytes")


def test_validate_file_metadata_empty_file():
    with pytest.raises(EmptyFileError, match="is empty"):
        validate_file_metadata("test.txt", b"")


def test_extract_valid_txt(sample_txt_bytes: bytes):
    result = extract_document("space.txt", sample_txt_bytes)
    assert result.filename == "space.txt"
    assert result.file_type == "txt"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Apollo 11" in result.pages[0].text
    assert result.total_characters > 100


def test_extract_whitespace_txt():
    with pytest.raises(NoExtractableTextError, match="contains no extractable text"):
        extract_document("blank.txt", b"   \n\n\t   \n")


def test_extract_valid_pdf(sample_pdf_bytes: bytes):
    result = extract_document("frameworks.pdf", sample_pdf_bytes)
    assert result.filename == "frameworks.pdf"
    assert result.file_type == "pdf"
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "FastAPI" in result.pages[0].text
    assert result.pages[1].page_number == 2
    assert "Streamlit" in result.pages[1].text
    assert result.total_characters > 100


def test_extract_scanned_image_only_pdf(image_only_pdf_bytes: bytes):
    with pytest.raises(NoExtractableTextError, match="contains no extractable text"):
        extract_document("scanned.pdf", image_only_pdf_bytes)


def test_extract_corrupted_pdf(corrupt_pdf_bytes: bytes):
    with pytest.raises(UnreadableDocumentError):
        extract_document("bad.pdf", corrupt_pdf_bytes)


def test_validate_file_metadata_file_too_large():
    from backend.app.services.exceptions import FileTooLargeError
    with pytest.raises(FileTooLargeError, match="exceeds the maximum limit"):
        validate_file_metadata("large.txt", b"A" * 105, max_bytes=100)

