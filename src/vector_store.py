"""
vector_store.py
Persistent ChromaDB vector store configured with cosine distance space
and strict document-level isolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

from src.text_chunker import PassageChunk


class VectorStoreError(Exception):
    """Base exception for ChromaDB vector store errors."""
    pass


@dataclass
class VectorSearchResult:
    """Represents a passage retrieved from the vector store with cosine distance and metadata."""
    chunk_id: int
    document_id: str
    source_filename: str
    page_number: int
    text: str
    distance: float  # Cosine distance: 0.0 (identical) to 2.0 (opposite)
    char_start: int = 0
    char_end: int = 0

    @property
    def similarity_score(self) -> float:
        """Normalized cosine similarity approximation: 1.0 - distance."""
        return max(0.0, round(1.0 - self.distance, 4))


_CHROMA_CLIENT_CACHE: Dict[str, Any] = {}


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage and document-isolated semantic queries."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "document_chunks",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._indexed_docs_cache = set()

    def _get_client_and_collection(self):
        """Lazy loader for persistent ChromaDB client and collection with process-level caching."""
        if self._collection is not None:
            return self._client, self._collection

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise VectorStoreError(
                "The 'chromadb' library is required for vector storage. "
                "Install it using: pip install chromadb"
            ) from e

        try:
            # Ensure storage directory exists
            abs_path = str(Path(self.persist_directory).resolve())
            Path(abs_path).mkdir(parents=True, exist_ok=True)

            if abs_path not in _CHROMA_CLIENT_CACHE:
                _CHROMA_CLIENT_CACHE[abs_path] = chromadb.PersistentClient(
                    path=abs_path,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

            self._client = _CHROMA_CLIENT_CACHE[abs_path]

            # Collection configured with cosine distance space:
            # smaller distance = closer / more similar.
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._client, self._collection
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB collection: {str(e)}") from e

    def is_document_indexed(self, document_id: str) -> bool:
        """Check if chunks for this document_id already exist in ChromaDB."""
        if not document_id:
            return False
        if document_id in self._indexed_docs_cache:
            return True
        _, collection = self._get_client_and_collection()
        try:
            existing = collection.get(where={"document_id": document_id}, limit=1)
            is_indexed = bool(existing and existing.get("ids"))
            if is_indexed:
                self._indexed_docs_cache.add(document_id)
            return is_indexed
        except Exception:
            return False

    def delete_document(self, document_id: str) -> None:
        """Remove all passages associated with a specific document_id."""
        if not document_id:
            return
        self._indexed_docs_cache.discard(document_id)
        _, collection = self._get_client_and_collection()
        try:
            collection.delete(where={"document_id": document_id})
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document '{document_id}' from ChromaDB: {str(e)}") from e

    def index_chunks(
        self,
        document_id: str,
        chunks: List[PassageChunk],
        embeddings: List[List[float]],
        replace: bool = False,
    ) -> int:
        """
        Store chunks and their externally computed embeddings in ChromaDB.
        Guarantees document isolation by filtering on document_id.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings provided.")

        self._indexed_docs_cache.add(document_id)
        _, collection = self._get_client_and_collection()

        # If replacing, clear existing chunks for this document
        if replace:
            self.delete_document(document_id)

        ids = [f"{document_id}_{chunk.chunk_id}" for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_id": int(chunk.chunk_id),
                "source_filename": str(chunk.source_filename),
                "page_number": int(chunk.page_number),
                "char_start": int(chunk.char_start),
                "char_end": int(chunk.char_end),
            }
            for chunk in chunks
        ]

        try:
            # Batch upsert into ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to index chunks into ChromaDB: {str(e)}") from e

    def query(
        self,
        document_id: str,
        query_embedding: List[float],
        top_k: int = 4,
        max_distance: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """
        Query passages for a specific document_id using cosine distance.
        Documents with different document_ids cannot contaminate the results.
        """
        if not document_id:
            return []

        _, collection = self._get_client_and_collection()

        try:
            # Query isolated by document_id metadata
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"document_id": document_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"Vector search failed: {str(e)}") from e

        results: List[VectorSearchResult] = []
        if not response or not response.get("ids") or not response["ids"][0]:
            return results

        doc_ids = response["ids"][0]
        documents = response["documents"][0] if response.get("documents") else []
        metadatas = response["metadatas"][0] if response.get("metadatas") else []
        distances = response["distances"][0] if response.get("distances") else []

        for i, chunk_key in enumerate(doc_ids):
            distance = float(distances[i]) if i < len(distances) else 1.0
            meta = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""

            # Filter by maximum distance if specified
            if max_distance is not None and distance > max_distance:
                continue

            results.append(
                VectorSearchResult(
                    chunk_id=meta.get("chunk_id", i + 1),
                    document_id=meta.get("document_id", document_id),
                    source_filename=meta.get("source_filename", "unknown"),
                    page_number=meta.get("page_number", 1),
                    text=text,
                    distance=round(distance, 4),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", len(text)),
                )
            )

        # Sort ascending by distance (closest first)
        results.sort(key=lambda r: r.distance)
        return results
