class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file is not an allowed extension (.txt, .pdf)."""
    pass


class EmptyFileError(DocumentProcessingError):
    """Raised when the uploaded file is zero bytes."""
    pass


class UnreadableDocumentError(DocumentProcessingError):
    """Raised when the document cannot be parsed or opened."""
    pass


class NoExtractableTextError(DocumentProcessingError):
    """Raised when the document contains pages but no extractable text (e.g. scanned image PDF)."""
    pass
