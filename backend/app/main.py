from fastapi import FastAPI, UploadFile, File, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from backend.app.schemas import (
    DocumentIndexResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
)
from backend.app.services.extractor import extract_document
from backend.app.services.chunker import create_chunks_from_document
from backend.app.services.state_manager import state_manager
from backend.app.services.exceptions import (
    UnsupportedFileTypeError,
    EmptyFileError,
    UnreadableDocumentError,
    NoExtractableTextError,
)

app = FastAPI(
    title="Document Question-Answering System Using RAG",
    description="A standalone RAG backend using PyMuPDF, scikit-learn TF-IDF, and FastAPI.",
    version="1.0.0",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint showing current active document state."""
    active_id = state_manager.get_active_document_id()
    session = state_manager.get_session(active_id) if active_id else None
    return HealthResponse(
        status="ok",
        active_document_id=active_id,
        total_indexed_chunks=len(session.chunks) if session else 0,
    )


@app.post(
    "/api/v1/documents/index",
    response_model=DocumentIndexResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def index_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=400, ge=100, le=2000),
    chunk_overlap: int = Query(default=50, ge=0, le=500),
    min_similarity_threshold: float = Query(default=0.15, ge=0.0, le=1.0),
):
    """Upload and index a TXT or text-based PDF document."""
    try:
        file_bytes = await file.read()
        extraction = extract_document(filename=file.filename or "", file_bytes=file_bytes)
        chunks = create_chunks_from_document(
            extraction=extraction,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise NoExtractableTextError(f"No usable text chunks could be created from '{file.filename}'.")

        doc_id = state_manager.store_document(
            extraction=extraction,
            chunks=chunks,
            min_similarity_threshold=min_similarity_threshold,
        )

        return DocumentIndexResponse(
            status="indexed",
            document_id=doc_id,
            filename=extraction.filename,
            file_type=extraction.file_type,
            total_pages=len(extraction.pages),
            total_chunks=len(chunks),
            total_characters=extraction.total_characters,
        )

    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EmptyFileError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except UnreadableDocumentError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except NoExtractableTextError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the document: {str(e)}",
        )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query_document(request: QueryRequest):
    """Query the indexed document with a natural-language question."""
    session = state_manager.get_session(request.document_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been indexed yet. Please upload and index a document before asking questions.",
        )

    response = session.rag_engine.answer_question(
        question=request.question,
        top_k=request.top_k,
        threshold=request.min_similarity_threshold,
    )
    return response


@app.post("/api/v1/documents/reset", tags=["Documents"])
async def reset_index():
    """Clear all in-memory document sessions."""
    state_manager.clear()
    return {"status": "cleared", "message": "All indexed document state has been cleared."}
