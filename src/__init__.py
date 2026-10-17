"""
Document Question-Answering System Using RAG - Core Source Package
"""

from src.document_loader import (
    load_document,
    ExtractedDocument,
    DocumentPage,
    DocumentValidationError,
    UnsupportedFileTypeError,
    EmptyFileError,
    UnreadableFileError,
    NoExtractableTextError,
)
from src.text_chunker import (
    TextChunker,
    PassageChunk,
    chunk_document,
)
from src.embedding_service import (
    EmbeddingService,
)
from src.vector_store import (
    ChromaVectorStore,
    VectorSearchResult,
)
from src.retriever import (
    SemanticRetriever,
    RetrievedPassage,
    TFIDFRetriever,
    RankedPassage,
)
from src.qa_engine import (
    QAEngine,
    QAResponse,
    ANSWER_NOT_FOUND_IN_DOCUMENT,
    validate_evidence,
)
from src.pipeline import DocumentQAPipeline
from src.config import (
    settings,
    Settings,
)

__all__ = [
    "load_document",
    "ExtractedDocument",
    "DocumentPage",
    "DocumentValidationError",
    "UnsupportedFileTypeError",
    "EmptyFileError",
    "UnreadableFileError",
    "NoExtractableTextError",
    "TextChunker",
    "PassageChunk",
    "chunk_document",
    "EmbeddingService",
    "ChromaVectorStore",
    "VectorSearchResult",
    "SemanticRetriever",
    "RetrievedPassage",
    "TFIDFRetriever",
    "RankedPassage",
    "QAEngine",
    "QAResponse",
    "ANSWER_NOT_FOUND_IN_DOCUMENT",
    "validate_evidence",
    "DocumentQAPipeline",
    "settings",
    "Settings",
]
