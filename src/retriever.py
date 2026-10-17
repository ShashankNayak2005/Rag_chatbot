"""
retriever.py
Semantic retrieval using local Sentence Transformers embeddings and ChromaDB vector store,
with backwards-compatible TF-IDF baseline support.
"""

from dataclasses import dataclass
from typing import List, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_chunker import PassageChunk
from src.document_loader import ExtractedDocument
from src.embedding_service import EmbeddingService
from src.vector_store import ChromaVectorStore, VectorSearchResult


@dataclass
class RetrievedPassage:
    """Represents a passage retrieved via semantic vector search with cosine distance and provenance."""
    chunk_id: int
    document_id: str
    source_filename: str
    page_number: int
    text: str
    distance: float  # Cosine distance: lower is more relevant (0 = identical)
    char_start: int = 0
    char_end: int = 0

    @property
    def similarity_score(self) -> float:
        """Normalized similarity approximation: 1.0 - distance."""
        return max(0.0, round(1.0 - self.distance, 4))


@dataclass
class RankedPassage:
    """Legacy/TF-IDF retrieved document passage representation."""
    chunk: PassageChunk
    similarity_score: float

    @property
    def distance(self) -> float:
        return max(0.0, round(1.0 - self.similarity_score, 4))


# Semantic domain mappings for query normalization and expansion
SEMANTIC_DOMAINS = {
    "founding": {
        "q_patterns": [
            r"\b(?:founded|founding|established|establishment|inception|incorporated)\b",
            r"\bwhen was (?:the )?(?:company|firm|datavex|it)\b",
            r"\b(?:what|which|in what|in which) year .*\b(?:start|founded|established|launch|begin)\b",
            r"\b(?:start|started|launch|launched|began) (?:operations|business)\b",
        ],
        "s_patterns": [
            r"\bfounded\b", r"\bestablished\b", r"\bstarted\b", r"\b(19\d\d|20\d\d)\b",
        ],
        "expansion": "founded establishment",
    },
    "location": {
        "q_patterns": [
            r"\b(?:headquarters|hq|main office|corporate office)\b",
            r"\b(?:located|location|based|situated|headquartered)\b",
            r"\bwhere is (?:the )?(?:company|firm|headquarters|office|datavex|it)\b",
            r"\b(?:which city|what city|in which city)\b",
        ],
        "s_patterns": [
            r"\bheadquarters\b", r"\blocat\b", r"\bbased\b", r"\bcity\b", r"\bhq\b", r"\boffice\b",
        ],
        "expansion": "headquarters location city",
    },
    "workforce": {
        "q_patterns": [
            r"\bhow many (?:employees|people|staff|workers|members|personnel)\b",
            r"\bhow many .* (?:work|employed|are there)\b",
            r"\b(?:headcount|staff|workforce|employees?|team size|worker count)\b",
            r"\bnumber of (?:employees|people|staff|workers)\b",
        ],
        "s_patterns": [
            r"\bemployees?\b", r"\bstaff\b", r"\bheadcount\b", r"\bworkforce\b",
        ],
        "expansion": "employees workforce staff headcount",
    },
    "services": {
        "q_patterns": [
            r"\b(?:main )?services?\b",
            r"\bsolutions?\b",
            r"\b(?:what does .* do|what are the services|what services)\b",
            r"\b(?:offerings?|products?|specialization)\b",
        ],
        "s_patterns": [
            r"\bservices?\b", r"\bsolutions?\b", r"\bproducts?\b", r"\bengineering\b", r"\bmain services\b",
        ],
        "expansion": "services solutions products",
    },
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercasing for consistent text matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


class SemanticRetriever:
    """
    Retrieves document passages using local Sentence Transformers embeddings
    and ChromaDB vector search with document isolation and distance thresholding.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        distance_threshold: float = 0.85,
        top_k: int = 4,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.distance_threshold = distance_threshold
        self.top_k = top_k

    def index_document(
        self,
        document: ExtractedDocument,
        chunks: List[PassageChunk],
        force_reindex: bool = False,
    ) -> int:
        """
        Embed and index document passages into ChromaDB.
        Reuses existing index if the document_id is already present.
        """
        if not chunks:
            return 0

        doc_id = document.document_id or chunks[0].document_id
        if not force_reindex and self.vector_store.is_document_indexed(doc_id):
            return len(chunks)

        # Batch embed chunk texts locally using sentence-transformers
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Upsert into ChromaDB with strict document_id isolation
        return self.vector_store.index_chunks(
            document_id=doc_id,
            chunks=chunks,
            embeddings=embeddings,
            replace=True,
        )

    def retrieve(
        self,
        document_id: str,
        query: str,
        top_k: Optional[int] = None,
        max_distance: Optional[float] = None,
    ) -> List[RetrievedPassage]:
        """
        Embed natural language query locally and retrieve nearest passages from ChromaDB.
        Filters out passages with cosine distance > max_distance.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        k = top_k or self.top_k
        cutoff = max_distance if max_distance is not None else self.distance_threshold

        # Generate query embedding locally with the exact same model
        query_embedding = self.embedding_service.embed_text(cleaned_query)

        # Query ChromaDB (isolated by document_id)
        search_results = self.vector_store.query(
            document_id=document_id,
            query_embedding=query_embedding,
            top_k=k,
            max_distance=cutoff,
        )

        return [
            RetrievedPassage(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                source_filename=r.source_filename,
                page_number=r.page_number,
                text=r.text,
                distance=r.distance,
                char_start=r.char_start,
                char_end=r.char_end,
            )
            for r in search_results
        ]


# -----------------------------------------------------------------------------
# Backwards-compatible TF-IDF Retriever
# -----------------------------------------------------------------------------
class TFIDFRetriever:
    """TF-IDF vector indexing and cosine ranking baseline."""

    def __init__(self, chunks: List[PassageChunk]):
        if not chunks:
            raise ValueError("TFIDFRetriever cannot be initialized with an empty chunks list.")

        self.chunks: List[PassageChunk] = chunks
        self.corpus: List[str] = [c.text for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            norm="l2",
        )

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        except ValueError:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),
                stop_words=None,
                sublinear_tf=True,
                norm="l2",
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def expand_query(self, query: str) -> str:
        q_lower = query.lower()
        expansions = []
        for domain in SEMANTIC_DOMAINS.values():
            if any(re.search(pat, q_lower) for pat in domain["q_patterns"]):
                expansions.append(domain["expansion"])
        if expansions:
            return f"{query} " + " ".join(expansions)
        return query

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> List[RankedPassage]:
        if not query or not query.strip():
            return []

        expanded_query = self.expand_query(query)
        q_vec = self.vectorizer.transform([expanded_query])

        if q_vec.nnz == 0:
            q_vec = self.vectorizer.transform([query])
            if q_vec.nnz == 0:
                return []

        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        ranked_indices = np.argsort(sims)[::-1]

        results: List[RankedPassage] = []
        for idx in ranked_indices[:top_k]:
            score = float(sims[idx])
            if score >= threshold and score > 0.0:
                results.append(
                    RankedPassage(
                        chunk=self.chunks[idx],
                        similarity_score=round(score, 4),
                    )
                )

        return results
