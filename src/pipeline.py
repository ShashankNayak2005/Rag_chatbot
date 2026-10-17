"""
pipeline.py
Unified Document Question-Answering RAG Pipeline.
Coordinates document loading, chunking, local embeddings, vector storage,
grounded retrieval, evidence validation, and natural language answer generation.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging

from src.document_loader import (
    load_document,
    ExtractedDocument,
    DocumentValidationError,
    EmptyFileError,
    UnsupportedFileTypeError,
    UnreadableFileError,
    NoExtractableTextError,
)
from src.text_chunker import chunk_document, PassageChunk, TextChunker
from src.embedding_service import EmbeddingService
from src.vector_store import ChromaVectorStore
from src.retriever import SemanticRetriever, RetrievedPassage
from src.qa_engine import (
    QAEngine,
    QAResponse,
    ANSWER_NOT_FOUND_IN_DOCUMENT,
    validate_evidence,
)
from src.config import settings

logger = logging.getLogger(__name__)


class DocumentQAPipeline:
    """
    High-level, end-to-end Document Question-Answering Pipeline.
    Encapsulates all RAG components into a simple, production-ready interface.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        distance_threshold: Optional[float] = None,
        top_k: int = 4,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        openai_client: Optional[Any] = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.top_k = top_k or settings.retrieval_top_k
        self.distance_threshold = (
            distance_threshold
            if distance_threshold is not None
            else settings.retrieval_distance_threshold
        )

        # 1. Local Embedding Model
        model_name = embedding_model or settings.embedding_model
        self.embedder = EmbeddingService(model_name=model_name)

        # 2. ChromaDB Vector Store
        p_dir = persist_directory or settings.chroma_persist_directory
        c_name = collection_name or settings.chroma_collection_name
        self.vector_store = ChromaVectorStore(
            persist_directory=p_dir,
            collection_name=c_name,
        )

        # 3. Semantic Retriever
        self.retriever = SemanticRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedder,
            distance_threshold=self.distance_threshold,
            top_k=self.top_k,
        )

        # 4. QA Engine with Answer Generation & Guardrails
        self.qa_engine = QAEngine(
            retriever=self.retriever,
            api_key=api_key or settings.active_api_key,
            base_url=base_url or settings.active_base_url,
            model=model or settings.active_model,
            distance_threshold=self.distance_threshold,
            openai_client=openai_client,
        )

        self._active_document: Optional[ExtractedDocument] = None
        self._active_chunks: List[PassageChunk] = []

    @property
    def active_document(self) -> Optional[ExtractedDocument]:
        """Return the currently loaded document metadata and structure."""
        return self._active_document

    @property
    def active_chunks(self) -> List[PassageChunk]:
        """Return the chunks for the active document."""
        return self._active_chunks

    def index_document(
        self,
        source: Union[str, Path, bytes],
        filename: Optional[str] = None,
        force_reindex: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> ExtractedDocument:
        """
        Load, validate, chunk, and index a document into ChromaDB.
        
        Args:
            source: File path (str/Path) or raw bytes.
            filename: Required if source is bytes.
            force_reindex: If True, overwrite any existing index for this document.
            chunk_size: Optional override for chunk size.
            chunk_overlap: Optional override for chunk overlap.
            
        Returns:
            ExtractedDocument containing text, pages, and metadata.
        """
        c_size = chunk_size or self.chunk_size
        c_overlap = chunk_overlap or self.chunk_overlap

        # Step 1: Load and validate document
        doc = load_document(source, filename=filename)

        # Step 2: Sentence-aware chunking with metadata preservation
        chunks = chunk_document(
            document=doc,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
        )

        # Step 3: Embed chunks locally and index into ChromaDB
        self.retriever.index_document(
            document=doc,
            chunks=chunks,
            force_reindex=force_reindex,
        )

        self._active_document = doc
        self._active_chunks = chunks
        return doc

    def answer_question(
        self,
        question: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        use_llm: bool = True,
        use_deepseek: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> QAResponse:
        """
        Query the active document and generate an evidence-grounded answer.
        
        Args:
            question: The user's natural language question.
            document_id: Document namespace to isolate retrieval. Defaults to active document.
            top_k: Number of candidate passages to retrieve.
            threshold: Optional distance threshold override.
            use_llm: Whether to generate natural language via Groq LLM or use extractive fallback.
            conversation_history: Optional list of past chat messages for multi-turn context.
            
        Returns:
            QAResponse with answer, evidence, citations, and status.
        """
        if use_deepseek is not None:
            use_llm = use_deepseek

        doc_id = document_id or (self._active_document.document_id if self._active_document else None)
        k = top_k or self.top_k
        t = threshold if threshold is not None else self.distance_threshold

        return self.qa_engine.answer_question(
            question=question,
            document_id=doc_id,
            top_k=k,
            threshold=t,
            use_llm=use_llm,
            conversation_history=conversation_history,
        )


__all__ = [
    "DocumentQAPipeline",
    "ANSWER_NOT_FOUND_IN_DOCUMENT",
]
