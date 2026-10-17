from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SourceReference(BaseModel):
    filename: str
    page_number: int
    chunk_id: int


class RetrievedChunk(BaseModel):
    chunk_id: int
    page_number: int
    similarity_score: float
    text: str


class DocumentIndexResponse(BaseModel):
    status: str = "indexed"
    document_id: str
    filename: str
    file_type: str
    total_pages: int
    total_chunks: int
    total_characters: int


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question to ask against the document.")
    document_id: Optional[str] = Field(default=None, description="Optional document ID. Uses active document if omitted.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top passages to retrieve.")
    min_similarity_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity cutoff to qualify as supported evidence."
    )

    @field_validator("question")
    @classmethod
    def validate_question_non_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be blank or contain only whitespace.")
        return stripped


class QueryResponse(BaseModel):
    question: str
    answer: str
    supporting_passage: Optional[str] = None
    source_reference: Optional[SourceReference] = None
    similarity_score: float
    retrieved_chunks: List[RetrievedChunk] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    active_document_id: Optional[str] = None
    total_indexed_chunks: int = 0
