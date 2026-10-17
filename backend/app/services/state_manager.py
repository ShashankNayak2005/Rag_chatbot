import uuid
from typing import Dict, List, Optional
from threading import Lock

from backend.app.services.extractor import ExtractionResult
from backend.app.services.chunker import DocumentChunk
from backend.app.services.rag_engine import RAGEngine


class DocumentSession:
    def __init__(
        self,
        document_id: str,
        extraction: ExtractionResult,
        chunks: List[DocumentChunk],
        rag_engine: RAGEngine,
    ):
        self.document_id = document_id
        self.extraction = extraction
        self.chunks = chunks
        self.rag_engine = rag_engine


class StateManager:
    """Thread-safe in-memory manager for document sessions and RAG indexes."""

    def __init__(self):
        self._lock = Lock()
        self._sessions: Dict[str, DocumentSession] = {}
        self._active_document_id: Optional[str] = None

    def store_document(
        self,
        extraction: ExtractionResult,
        chunks: List[DocumentChunk],
        min_similarity_threshold: float = 0.15,
    ) -> str:
        """Create a new session, fit the RAG engine, and store in-memory."""
        doc_id = str(uuid.uuid4())[:8]
        engine = RAGEngine(chunks=chunks, min_similarity_threshold=min_similarity_threshold)
        session = DocumentSession(
            document_id=doc_id,
            extraction=extraction,
            chunks=chunks,
            rag_engine=engine,
        )

        with self._lock:
            self._sessions[doc_id] = session
            self._active_document_id = doc_id

        return doc_id

    def get_session(self, document_id: Optional[str] = None) -> Optional[DocumentSession]:
        """Retrieve a session by ID, or fallback to the active session."""
        with self._lock:
            target_id = document_id or self._active_document_id
            if target_id and target_id in self._sessions:
                return self._sessions[target_id]
            return None

    def get_active_document_id(self) -> Optional[str]:
        with self._lock:
            return self._active_document_id

    def clear(self):
        """Clear all loaded sessions and active state."""
        with self._lock:
            self._sessions.clear()
            self._active_document_id = None


# Global singleton instance
state_manager = StateManager()
