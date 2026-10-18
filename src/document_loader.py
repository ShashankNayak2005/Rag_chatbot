"""
document_loader.py
Extracts and validates plain-text (TXT) and text-based PDF documents using PyMuPDF.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, BinaryIO
import os
import pymupdf as fitz


class DocumentValidationError(Exception):
    """Base exception for document validation and extraction errors."""
    pass


class UnsupportedFileTypeError(DocumentValidationError):
    """Raised when file extension is not .txt or .pdf."""
    pass


class EmptyFileError(DocumentValidationError):
    """Raised when the provided file contains 0 bytes."""
    pass


class FileTooLargeError(DocumentValidationError):
    """Raised when the provided file exceeds the maximum allowed size limit."""
    pass


class UnreadableFileError(DocumentValidationError):
    """Raised when a PDF or text file cannot be parsed or decoded."""
    pass


class NoExtractableTextError(DocumentValidationError):
    """Raised when a valid file contains 0 extractable text characters (e.g. scanned image PDF)."""
    pass


@dataclass
class DocumentPage:
    """Represents text extracted from a discrete page of a document."""
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    """Represents a validated, extracted document with page boundaries."""
    filename: str
    file_type: str
    pages: List[DocumentPage]
    total_characters: int
    document_id: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


ALLOWED_EXTENSIONS = {".txt", ".pdf"}



def _read_bytes_and_filename(
    source: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
) -> tuple[bytes, str]:
    """Helper to convert various source types into raw bytes and a sanitized filename."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Document file not found: {source}")
        fname = filename or p.name
        content = p.read_bytes()
        return content, fname

    if isinstance(source, bytes):
        fname = filename or "uploaded_document"
        return source, fname

    # File-like object (e.g. Streamlit UploadedFile or BytesIO)
    if hasattr(source, "read"):
        fname = filename or getattr(source, "name", "uploaded_document")
        current_pos = getattr(source, "tell", lambda: 0)()
        try:
            content = source.read()
        finally:
            if hasattr(source, "seek"):
                source.seek(current_pos)
        if isinstance(content, str):
            content = content.encode("utf-8")
        return content, fname

    raise TypeError(f"Unsupported document source type: {type(source)}")


def validate_file_metadata(
    filename: str,
    file_bytes: bytes,
    max_bytes: Optional[int] = None,
) -> str:
    """
    Validate file extension, ensure non-zero byte size, and check max file size limit.
    Returns normalized extension (e.g. '.txt', '.pdf').
    """
    if not filename:
        raise UnsupportedFileTypeError("Filename cannot be empty.")

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Only plain text (.txt) and text-based PDF (.pdf) documents are supported."
        )

    if len(file_bytes) == 0:
        raise EmptyFileError(f"The document '{filename}' is empty (0 bytes).")

    if max_bytes is not None and len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise FileTooLargeError(
            f"The document '{filename}' is {size_mb:.2f} MB, which exceeds the maximum allowed limit of {limit_mb:.2f} MB."
        )

    return ext


def extract_from_txt(filename: str, file_bytes: bytes) -> ExtractedDocument:
    """Extract and validate text from a plain text document."""
    try:
        decoded_text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            decoded_text = file_bytes.decode("latin-1")
        except Exception as e:
            raise UnreadableFileError(f"Failed to decode text file '{filename}': {str(e)}") from e

    # Normalize line endings and strip leading/trailing whitespace
    cleaned_text = decoded_text.replace("\r\n", "\n").strip()
    if not cleaned_text:
        raise NoExtractableTextError(f"The text file '{filename}' contains no readable text.")

    pages = [DocumentPage(page_number=1, text=cleaned_text)]
    return ExtractedDocument(
        filename=filename,
        file_type="txt",
        pages=pages,
        total_characters=len(cleaned_text),
    )



def extract_from_pdf(filename: str, file_bytes: bytes) -> ExtractedDocument:
    """Extract and validate text from a PDF document using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise UnreadableFileError(f"Failed to open/parse PDF document '{filename}': {str(e)}") from e

    try:
        if doc.page_count == 0:
            raise NoExtractableTextError(f"The PDF '{filename}' contains 0 pages.")

        pages: List[DocumentPage] = []
        total_chars = 0

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            raw_text = page.get_text() or ""
            cleaned = raw_text.strip()
            if cleaned:
                pages.append(DocumentPage(page_number=page_idx + 1, text=cleaned))
                total_chars += len(cleaned)

        if total_chars == 0 or len(pages) == 0:
            raise NoExtractableTextError(
                f"The PDF '{filename}' contains no extractable text. Scanned or image-only documents require OCR (Optical Character Recognition) to be processed."
            )

        return ExtractedDocument(
            filename=filename,
            file_type="pdf",
            pages=pages,
            total_characters=total_chars,
        )
    finally:
        doc.close()


def load_document(
    source: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> ExtractedDocument:
    """
    Main entrypoint to validate and extract text from a TXT or PDF document.
    
    Accepts:
        - file path (str or Path)
        - raw bytes
        - binary file stream (e.g., Streamlit UploadedFile)
    """
    import hashlib
    from src.config import get_settings

    if max_bytes is None:
        max_bytes = get_settings().max_file_size_bytes

    file_bytes, fname = _read_bytes_and_filename(source, filename)
    ext = validate_file_metadata(fname, file_bytes, max_bytes=max_bytes)

    if ext == ".txt":
        doc = extract_from_txt(fname, file_bytes)
    elif ext == ".pdf":
        doc = extract_from_pdf(fname, file_bytes)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file extension: {ext}")

    doc.document_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    return doc

