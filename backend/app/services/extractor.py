from dataclasses import dataclass
from typing import List
import os
import pymupdf as fitz

from backend.app.services.exceptions import (
    UnsupportedFileTypeError,
    EmptyFileError,
    UnreadableDocumentError,
    NoExtractableTextError,
)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@dataclass
class ExtractedPage:
    page_number: int
    text: str


@dataclass
class ExtractionResult:
    filename: str
    file_type: str
    pages: List[ExtractedPage]
    total_characters: int


def validate_file_metadata(filename: str, file_bytes: bytes) -> str:
    """Validate filename extension and non-empty byte payload. Returns lowercase extension."""
    if not filename:
        raise UnsupportedFileTypeError("Filename cannot be empty.")
    
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Only .txt and .pdf files are accepted."
        )
    
    if len(file_bytes) == 0:
        raise EmptyFileError(f"The file '{filename}' is empty (0 bytes).")
        
    return ext


def extract_text_from_txt(filename: str, file_bytes: bytes) -> ExtractionResult:
    """Extract and validate text from a plain text document."""
    try:
        decoded_text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded_text = file_bytes.decode("latin-1")
        except Exception as e:
            raise UnreadableDocumentError(f"Could not decode TXT file '{filename}': {str(e)}") from e

    cleaned_text = decoded_text.strip()
    if not cleaned_text:
        raise NoExtractableTextError(f"The text file '{filename}' contains no extractable text.")

    # TXT documents are treated as a single page or segmented by form feeds if present
    pages = [ExtractedPage(page_number=1, text=cleaned_text)]
    return ExtractionResult(
        filename=filename,
        file_type="txt",
        pages=pages,
        total_characters=len(cleaned_text),
    )


def extract_text_from_pdf(filename: str, file_bytes: bytes) -> ExtractionResult:
    """Extract and validate text page-by-page from a PDF document using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise UnreadableDocumentError(f"Failed to open/parse PDF file '{filename}': {str(e)}") from e

    try:
        if doc.page_count == 0:
            raise NoExtractableTextError(f"The PDF file '{filename}' contains 0 pages.")

        pages: List[ExtractedPage] = []
        total_chars = 0

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            page_text = page.get_text() or ""
            cleaned = page_text.strip()
            if cleaned:
                pages.append(ExtractedPage(page_number=page_idx + 1, text=cleaned))
                total_chars += len(cleaned)

        if total_chars == 0 or len(pages) == 0:
            raise NoExtractableTextError(
                f"The PDF '{filename}' contains no extractable text. Scanned or image-only documents require OCR (Optical Character Recognition) to be processed."
            )

        return ExtractionResult(
            filename=filename,
            file_type="pdf",
            pages=pages,
            total_characters=total_chars,
        )
    finally:
        doc.close()


def extract_document(filename: str, file_bytes: bytes) -> ExtractionResult:
    """High-level entrypoint to validate and extract text from supported documents."""
    ext = validate_file_metadata(filename, file_bytes)
    if ext == ".txt":
        return extract_text_from_txt(filename, file_bytes)
    elif ext == ".pdf":
        return extract_text_from_pdf(filename, file_bytes)
    else:
        raise UnsupportedFileTypeError(f"Unsupported extension: {ext}")
