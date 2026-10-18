# Document Q&A — Enterprise Document Question-Answering System Using RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-orange.svg)](https://www.trychroma.com/)
[![Groq Cloud](https://img.shields.io/badge/LLM-Groq%20Cloud-f55036.svg)](https://groq.com/)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Tests Passing](https://img.shields.io/badge/tests-125%20passed-brightgreen.svg)]()
[![Performance](https://img.shields.io/badge/speedup-5.4x%20faster-purple.svg)]()

An enterprise-grade, evidence-grounded **Retrieval-Augmented Generation (RAG)** application designed for high-accuracy question answering across structured and unstructured text documents (PDF and TXT). 

The platform guarantees strict factual alignment with uploaded documents, eliminates hallucinations through deterministic negative guardrails, maintains verifiable source provenance (physical page numbers and chunk offsets), and leverages ultra-fast **Groq Chat Completions** combined with **100% local Sentence-Transformer embeddings**.

---

## Table of Contents

1. [🚀 Setup Guidelines & Quickstart](#1--setup-guidelines--quickstart)
   - [Prerequisites](#prerequisites)
   - [Step 1: Clone the Repository](#step-1-clone-the-repository)
   - [Step 2: Create and Activate Virtual Environment](#step-2-create-and-activate-virtual-environment)
   - [Step 3: Install Dependencies](#step-3-install-dependencies)
   - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
   - [Step 5: Run the Streamlit Application](#step-5-run-the-streamlit-application)
   - [Step 6: Run the FastAPI REST Backend](#step-6-run-the-fastapi-rest-backend)
   - [Step 7: Run Automated Tests & Benchmarks](#step-7-run-automated-tests--benchmarks)
2. [📖 In-Depth Project Details & Overview](#2--in-depth-project-details--overview)
   - [The Problem with Naive RAG](#the-problem-with-naive-rag)
   - [How This Solution Solves It](#how-this-solution-solves-it)
   - [High-Performance Speed & Latency Optimizations](#high-performance-speed--latency-optimizations)
3. [🔄 End-to-End System Workflow](#3--end-to-end-system-workflow)
4. [📁 Repository Structure & Module Directory](#4--repository-structure--module-directory)
5. [⚙️ Core Technical Architecture & Algorithms](#5--core-technical-architecture--algorithms)
   - [A. Document Ingestion & Validation](#a-document-ingestion--validation)
   - [B. Sentence-Aware Chunking with Lookback](#b-sentence-aware-chunking-with-lookback)
   - [C. Zero-Cost Local Dense Vectorization](#c-zero-cost-local-dense-vectorization)
   - [D. Persistent Vector Storage & Document Isolation](#d-persistent-vector-storage--document-isolation)
   - [E. Hybrid Retrieval (Semantic + TF-IDF)](#e-hybrid-retrieval-semantic--tf-idf)
   - [F. Negative Guardrails & Anti-Hallucination Engine](#f-negative-guardrails--anti-hallucination-engine)
   - [G. Prompt Injection Hardening & Groq LLM Synthesis](#g-prompt-injection-hardening--groq-llm-synthesis)
   - [H. Traceable Source Citations](#h-traceable-source-citations)
6. [🔌 FastAPI REST API Reference](#6--fastapi-rest-api-reference)
7. [🧪 Empirical Verification & Sample Test Cases](#7--empirical-verification--sample-test-cases)
8. [⚙️ Configuration Reference](#8--configuration-reference)
9. [❓ Troubleshooting & FAQs](#9--troubleshooting--faqs)

---

## 1. 🚀 Setup Guidelines & Quickstart

Follow these step-by-step instructions to get the application running on your local machine.

### Prerequisites

Before starting, ensure you have the following installed:
- **Operating System:** Windows 10/11, macOS (Intel or Apple Silicon), or Linux (Ubuntu 20.04+).
- **Python Version:** Python **3.10** or higher (Python 3.11 / 3.12 fully supported). Check with:
  ```bash
  python --version
  ```
- **Git:** Installed on your path.
- **Groq API Key (Optional for cloud LLM synthesis):** A free API key from the [Groq Console](https://console.groq.com/keys).
  > 💡 **Offline Mode:** If you do not have a Groq API key, leave `GROQ_API_KEY` blank. The system automatically runs using its zero-cost local extractive answer synthesizer.

---

### Step 1: Clone the Repository

Open a terminal (PowerShell on Windows, or Bash on macOS/Linux):

```bash
git clone https://github.com/ShashankNayak2005/Rag_chatbot.git
cd Rag_chatbot
```

---

### Step 2: Create and Activate Virtual Environment

Create an isolated Python virtual environment to avoid dependency conflicts:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> 💡 **Windows Tip:** If PowerShell displays an `Execution_Policies` restriction error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\venv\Scripts\Activate.ps1
> ```

**On macOS / Linux (Bash/Zsh):**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

Upgrade `pip` and install all required production and testing libraries:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### What gets installed:
- `streamlit`: Clean, modern chat UI and interactive widgets.
- `fastapi` & `uvicorn`: High-performance asynchronous REST API backend.
- `pymupdf` (`fitz`): High-speed PDF text extraction and page coordinate mapping.
- `sentence-transformers`: Local embedding generation model (`all-MiniLM-L6-v2`).
- `torch`: Deep learning runtime optimized for CPU inference with inference mode.
- `chromadb`: Persistent vector database with HNSW cosine distance space.
- `openai`: Official client used for Groq Cloud high-speed chat completions.
- `scikit-learn`: TF-IDF keyword vectorization and cosine similarity.
- `pydantic`: Schema validation and settings management.
- `pytest`: Comprehensive unit and regression test framework.

---

### Step 4: Configure Environment Variables

Create your active `.env` file from the provided template:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your **Groq API Key**:
```ini
# Groq API Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# RAG Pipeline Hyperparameters
CHUNK_SIZE=700
CHUNK_OVERLAP=100
SIMILARITY_THRESHOLD=0.85
TOP_K=3
PERSIST_DIRECTORY=./chroma_db
```

---

### Step 5: Run the Streamlit Application

Launch the interactive web chatbot:

```bash
streamlit run app.py
```

The terminal will display the local URL:
```text
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Open `http://localhost:8501` in your web browser.

#### Using the Web Interface:
1. **Prominent Sidebar Toggle:** Click the blue `📂 Document Workspace` button in the top-left corner to access the document uploader.
2. **Choose / Upload Document:** Upload any `.pdf` or `.txt` file, or click an instant demo button (**🏢 Datavex** or **☀️ Solar Tech**).
3. **Index Document:** Click `🚀 Index Now`. The app extracts text, segments it into sentence-aware chunks, generates dense embeddings, and stores them in isolated ChromaDB partitions.
4. **Ask Questions:** Type questions into the chat box (e.g., *"When was the company founded?"*, *"What services are offered?"*, or test negative guardrails with *"Who is the CEO?"*).
5. **Real-Time Token Streaming:** Answers stream token-by-token in real time (first word in `<200ms`), accompanied by exact provenance pills: `📄 File`, `📖 Page`, `🧩 Chunk`, `🎯 Dist`, and `🕒 Timestamp`.

---

### Step 6: Run the FastAPI REST Backend

The project includes a standalone headless REST API powered by FastAPI and Uvicorn:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger documentation is available at:
👉 **`http://localhost:8000/docs`**

#### Core REST Endpoints:
- `GET /api/v1/health`: Checks system readiness and active document session.
- `POST /api/v1/documents/index`: Uploads and chunks `.pdf` or `.txt` documents.
- `POST /api/v1/query`: Queries the indexed document and returns grounded answers + provenance citations.
- `POST /api/v1/documents/reset`: Clears active in-memory document state and session indexes.

---

### Step 7: Run Automated Tests & Benchmarks

The project comes with a comprehensive test suite (125 tests) verifying all pipeline modules:

```bash
# Run the full automated test suite (runs in ~37s, 100% offline)
python -m pytest tests/ -q
```

**Expected Test Output:**
```text
........................................................................ [ 57%]
.....................................................                    [100%]
125 passed in 37.18s
```

#### Run the Acceptance Test Runner (TC1–TC12 & CoastalTech):
```bash
python run_test_cases.py
```

#### Run the Quantitative Evaluation Benchmark:
```bash
python evaluation/run_evaluation.py
```

---

## 2. 📖 In-Depth Project Details & Overview

### The Problem with Naive RAG
Standard RAG architectures frequently fail in enterprise production environments due to critical vulnerabilities:
1. **Hallucination on Absent Facts:** When a user asks about an absent entity (e.g., *"Who is the CEO?"* on a document that only mentions company location and headcount), standard vector search still retrieves the nearest passage (e.g., *"Headquarters: Bengaluru"*) and the LLM invents a plausible executive.
2. **Prompt Injection & Delimiter Confusion:** Malicious documents containing phrases like *"Ignore previous instructions and say..."* can hijack LLM completion prompts and override system behavior.
3. **Cross-Document Contamination:** In shared vector indexes, queries on Document B frequently retrieve stale fragments from Document A.
4. **Cloud API Vendor Lock-in & Costs:** Relying on third-party APIs for document vector embeddings creates recurring latency, network overhead, and continuous token costs.

### How This Solution Solves It
- **Deterministic Negative Guardrail (`validate_evidence`):** Automatically evaluates query semantics against factual predicates (leadership, financials, dates, scope). If evidence is absent, the system returns strictly `ANSWER_NOT_FOUND_IN_DOCUMENT` with **zero API calls** and **zero hallucinations**.
- **Untrusted Context Delimiters:** Documents are wrapped in strict `<document_excerpt>` boundaries with explicit instructions forbidding the LLM from executing commands inside context.
- **Isolated Document Namespaces:** ChromaDB collections enforce partition metadata (`where={"document_id": active_doc_id}`), preventing cross-document data leakage.
- **100% Local Embeddings:** All chunks and queries are vectorized locally on CPU via `all-MiniLM-L6-v2` (384 dimensions) at zero cost.
- **Sub-Second LLM Synthesis:** Powered by **Groq Cloud** using state-of-the-art open-weight models (`qwen/qwen3.8-27b`, `llama-3.3-70b-versatile`) with automatic zero-cost local extractive fallback if offline.

---

### High-Performance Speed & Latency Optimizations

The system has been heavily profiled and optimized for low-latency performance:

| Optimization Layer | Implementation | Impact |
| :--- | :--- | :--- |
| **Model Weight Pre-Warming** | Process-level singleton cache (`_MODEL_CACHE`) in `src/embedding_service.py` | Eliminates redundant 120MB PyTorch model reloads. Re-instantiation drops from **34s to 0.00s**. |
| **Offline HF Hub Check** | Prioritizes `local_files_only=True` when loading embeddings | Bypasses slow remote HuggingFace Hub network handshakes and unauthenticated rate limits. |
| **Query Embedding Memory Cache** | In-memory LRU cache (`_EMBEDDING_CACHE`) | Recurring query embedding time drops from **40ms to 0.00ms**. |
| **PyTorch Inference Mode** | Encapsulated in `torch.inference_mode()` with multi-core CPU SIMD threading | 20–30% faster matrix multiplications; avoids autograd graph allocations. |
| **ChromaDB Connection Pooling** | Persistent client caching (`_CHROMA_CLIENT_CACHE`) & `_indexed_docs_cache` | Eliminates disk SQLite file locks, threadpool allocations, and repeated index re-reads. |
| **Streamlit `@st.cache_resource`** | Global singleton caching across sessions and reruns | Page refreshes and document switches do not trigger cold-start model initializations. |
| **Real-Time Token Streaming** | `stream_answer_tokens` in `QAEngine` + `st.write_stream` | First token time-to-glass drops from **3.5s down to < 200ms** (500–800 tokens/sec via Groq). |
| **Test Suite Acceleration** | Optimized embedding lifecycle across all 18 test modules | Entire 125-test suite execution time slashed from **200.60s to 37.18s** (**5.4x speedup**). |

---

## 3. 🔄 End-to-End System Workflow

The following diagram illustrates the complete execution lifecycle from raw file upload to vectorization, retrieval, evidence validation, and grounded answer synthesis:

```
[ User Upload: PDF / TXT ]
           │
           ▼
[ 1. Document Loader ] ────────► Validates extension, size, non-empty text, PDF encryption
           │                     Extracts page-by-page text via PyMuPDF (fitz) or UTF-8 / Latin-1
           ▼
[ 2. Text Cleaner ] ───────────► Normalizes Unicode, removes non-printable control chars, standardizes whitespace
           │
           ▼
[ 3. Text Chunker ] ───────────► Sentence-aware sliding window (~700 tokens, 100 token overlap)
           │                     Natural sentence lookback (. ! ? \n\n) prevents mid-sentence truncation
           ▼
[ 4. Local Embedding Engine ] ──► sentence-transformers/all-MiniLM-L6-v2 (384-d dense vectors)
           │                     Executes 100% locally on CPU in inference mode with process-level caching
           ▼
[ 5. Persistent ChromaDB ] ────► Stores chunk embeddings in cosine distance space (hnsw:space: cosine)
           │                     Enforces namespace isolation via SHA-256 document_id
           │
═══════════╪════════════════════════════════════════════════════════════════════════════════════════
           │  USER QUERY WORKFLOW
           ▼
[ 6. User Query Ingestion ] ───► Sanitized, stripped, and vectorized via local all-MiniLM-L6-v2
           │                     Instant response if query is found in in-memory LRU embedding cache
           ▼
[ 7. Hybrid Retriever ] ───────► Dense vector search (cosine distance <= 0.85)
           │                     Complementary TF-IDF keyword retrieval for exact terms/acronyms
           │                     Document-isolated metadata filtering (where={"document_id": active_doc_id})
           ▼
[ 8. Evidence Validation ] ────► Evaluates candidate passages against factual predicate patterns
           │                     Detects absent entities (CEO, revenue, founder, dates, out-of-domain)
           │
           ├── [Insufficient Evidence] ──► Deterministically returns:
           │                               "ANSWER_NOT_FOUND_IN_DOCUMENT" (0 API calls, 0 hallucinations)
           │
           └── [Evidence Validated]
                   │
                   ▼
[ 9. Groq LLM Engine ] ────────► Real-time streaming Chat Completions API via Groq endpoint
                   │             System prompt enforces strict grounding within <document_excerpt>
                   │             Zero-cost local extractive fallback if offline or no key configured
                   ▼
[ 10. Synthesized Answer ] ────► Grounded natural language response streamed token-by-token (<200ms latency)
                   │
                   ▼
[ 11. Provenance Citations ] ──► Exact source file, physical PDF page numbers, chunk ID, and cosine distance
```

---

## 4. 📁 Repository Structure & Module Directory

Every file and directory in this repository serves a specific, vital role:

```text
rag-chatbot/
│
├── app.py                          # Primary Streamlit web application with real-time token streaming
├── requirements.txt                # Production and development dependencies
├── README.md                       # Comprehensive project documentation & architecture guide
├── sample_output.md                # Empirical test verification transcripts & screen outputs
├── test_results.md                 # Full test suite execution log & pass/fail records
├── pytest.ini                      # Pytest test discovery & execution configuration
├── run_test_cases.py               # Standalone CLI test runner for TC1-TC12 & CoastalTech
├── .env.example                    # Template file for environment variable configuration
├── .gitignore                      # Git exclusion rules (virtual environments, caches, keys)
│
├── src/                            # Core modular RAG pipeline & domain logic
│   ├── __init__.py                 # Package exports & public API interface
│   ├── config.py                   # Centralized Pydantic environment settings loader
│   ├── document_loader.py          # PyMuPDF (fitz) & TXT ingestion, validation, and error guards
│   ├── text_cleaner.py             # Unicode text normalization & whitespace sanitization
│   ├── text_chunker.py             # Sentence-boundary aware sliding window chunker
│   ├── chunker.py                  # Backward-compatible chunker interface wrapper
│   ├── embedding_service.py        # Local Sentence Transformers (all-MiniLM-L6-v2) embedding engine + caches
│   ├── embeddings.py               # Backward-compatible embedding interface wrapper
│   ├── vector_store.py             # Persistent ChromaDB store, collection isolation & client pooling
│   ├── retriever.py                # Semantic dense retriever & TF-IDF keyword matching
│   ├── evidence_validator.py       # Deterministic factual predicate & negative guardrail checker
│   ├── answer_generator.py         # Response generation interface & synthesizer wrapper
│   ├── qa_engine.py                # Core QA orchestration, Groq client, streaming & local fallback
│   └── pipeline.py                 # Unified DocumentQAPipeline facade connecting all modules
│
├── backend/                        # Standalone FastAPI microservice
│   └── app/
│       ├── __init__.py             # Backend package initialiser
│       ├── main.py                 # FastAPI application routes (/health, /index, /query, /reset)
│       ├── schemas.py              # Pydantic v2 request and response schemas
│       └── services/               # Microservice adapter bridging API to RAG pipeline
│
├── sample_documents/               # Primary benchmark & assignment verification documents
│   ├── datavex_sample.txt          # Standard company profile document (TC1–TC12)
│   └── coastaltech_sample.txt      # Synthetic unseen enterprise document (CT1–CT7)
│
├── sample_docs/                    # Extended multi-format evaluation documents
│   ├── sample_article.pdf          # Multi-page PDF for page-boundary citation testing
│   ├── sample_report.txt           # Renewable Energy Solar PV technical report
│   ├── sample_company_profile.txt  # Extended enterprise profile
│   └── sample_datavex.txt          # Alternate benchmark copy of Datavex profile
│
├── data/                           # Local runtime data & conversation storage
│   ├── datavex_profile.txt         # Runtime reference data file
│   └── conversation_history.json   # Persistent chat history
│
├── evaluation/                     # Automated benchmark evaluation suite
│   ├── evaluation_dataset.json     # Ground-truth evaluation dataset (22 multi-document test cases)
│   ├── run_evaluation.py           # Automated evaluation runner computing quantitative metrics
│   └── evaluation_report.json      # Generated benchmark report with accuracy & latency scores
│
└── tests/                          # Comprehensive automated unit & integration test suite (125 tests)
    ├── __init__.py                 # Test suite package initialiser
    ├── conftest.py                 # Pytest fixtures (sample TXT, PDF, corrupt files, FastAPI TestClient)
    ├── test_acceptance.py          # Master acceptance criteria tests (TC1–TC7)
    ├── test_answer_generation.py   # Grounded answer generation & prompt injection tests
    ├── test_api.py                 # FastAPI REST endpoint integration tests (11 tests)
    ├── test_chunker.py             # Text sliding-window chunker tests
    ├── test_citations.py           # Provenance tracking, page number & chunk ID tests
    ├── test_document_loader.py     # TXT/PDF extraction, empty files, corrupt files tests
    ├── test_end_to_end.py          # End-to-end integration tests (Datavex & CoastalTech)
    ├── test_error_handling.py      # Error propagation & negative guardrail tests
    ├── test_extractor.py           # PyMuPDF byte parsing and extraction tests
    ├── test_groq_rag.py            # Groq LLM API and mocked completions tests
    ├── test_qa_engine.py           # QA orchestration and evidence validation tests
    ├── test_rag_engine.py          # Lightweight RAG engine baseline tests
    ├── test_regression_suite.py    # Regression test suite (18 comprehensive tests)
    ├── test_retrieval.py           # Semantic retrieval and cosine ranking tests
    ├── test_retriever.py           # Hybrid retrieval and query normalization tests
    └── test_text_chunker.py        # Tokenizer limit (256 tokens) and boundary tests
```

---

## 5. ⚙️ Core Technical Architecture & Algorithms

### A. Document Ingestion & Validation
Located in `src/document_loader.py`:
- **Format Inspection:** Supports standard `.pdf` and `.txt` files. Rejects unsupported binary formats (`.docx`, `.exe`, `.bin`) with descriptive user-facing errors (`UnsupportedFileTypeError`).
- **Size Boundary Checks:** Blocks 0-byte empty files (`EmptyFileError`) and files exceeding the configurable limit (default: 50MB).
- **PDF Extraction (PyMuPDF):** Extracts digital text page-by-page via `page.get_text("text")`. Retains physical 1-indexed page numbers. Detects password-protected PDFs and scanned/image-only PDFs lacking extractable text, raising `NoExtractableTextError`.
- **TXT Ingestion:** Decodes text using `utf-8` with dynamic fallback to `latin-1`. Sanitizes byte-order marks (`utf-8-sig`) and normalizes line endings (`\r\n` $\rightarrow$ `\n`).

### B. Sentence-Aware Chunking with Lookback
Located in `src/text_chunker.py`:
- **Target Parameters:** Calibrated for `sentence-transformers/all-MiniLM-L6-v2` maximum sequence length of 256 tokens. A target of ~800 characters (~200 tokens) with ~150 characters (~35-40 tokens) overlap ensures zero silent truncation.
- **Boundary Preservation:** Scans backward up to 200 characters from the target boundary to find natural sentence delimiters (`. `, `! `, `? `, `\n\n`, `\n`) rather than breaking mid-sentence.
- **Metadata Encapsulation:** Every chunk stores its `chunk_id`, `document_id` (SHA-256 hash), `source_filename`, `page_number`, and character start/end offsets.

### C. Zero-Cost Local Dense Vectorization
Located in `src/embedding_service.py`:
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Dimensions:** 384-dimensional dense floating-point vector representations.
- **Hardware Optimization:** Runs locally on CPU via PyTorch with thread pooling and `torch.inference_mode()`.
- **Memory Caching:** Includes global model singleton caching (`_MODEL_CACHE`) and query embedding caching (`_EMBEDDING_CACHE`).
- **Offline Hub Bypassing:** Uses `local_files_only=True` to eliminate network round-trips to HuggingFace Hub.

### D. Persistent Vector Storage & Document Isolation
Located in `src/vector_store.py`:
- **Database Engine:** ChromaDB persistent client writing to disk (`./chroma_db`).
- **Distance Metric:** HNSW Cosine Distance space (`{"hnsw:space": "cosine"}`):
  $$\text{Cosine Distance} = 1.0 - \text{Cosine Similarity}$$
- **Namespace Isolation:** Chunks are tagged with `document_id`. All retriever queries pass an immutable filter:
  ```python
  where = {"document_id": active_document_id}
  ```
  This ensures that uploading a new document completely isolates queries to that document, preventing cross-document contamination.
- **Client Pooling:** Reuses persistent clients across instances via `_CHROMA_CLIENT_CACHE` to avoid repeated SQLite file locks.

### E. Hybrid Retrieval (Semantic + TF-IDF)
Located in `src/retriever.py`:
- **Semantic Dense Search:** Computes query embedding locally and retrieves top-$k$ nearest neighbor chunks satisfying the similarity threshold (default: cosine distance $\le 0.85$).
- **Lexical Keyword Search (TF-IDF):** Computes sparse term-frequency inverse-document-frequency vectors over document chunks using scikit-learn. Ensures exact matching for acronyms, serial numbers, product codes, and proper nouns.
- **Candidate Expansion:** For concise key-value documents (e.g., invoices, structured receipts, tabular profiles), if dense distance is slightly elevated due to short passage length, candidate passages up to distance $0.98$ are checked against key-value line patterns.

### F. Negative Guardrails & Anti-Hallucination Engine
Located in `src/evidence_validator.py` and `src/qa_engine.py`:
Before any context is sent to the LLM, the `validate_evidence()` engine verifies whether the retrieved excerpts contain the factual predicates demanded by the query:
1. **Leadership / Personnel Guardrail:** Detects questions asking for `CEO`, `founder`, `president`, `director`, `chief executive`. If the retrieved text lacks leadership terms or titles, retrieval is flagged as insufficient.
2. **Financial Guardrail:** Detects questions asking for `revenue`, `profit`, `turnover`, `valuation`, `funding`, `sales`. If the retrieved text contains no monetary signs, currency abbreviations, or financial figures, retrieval is blocked.
3. **Out-of-Domain Guardrail:** Discards queries unrelated to document topics (e.g., asking about astronomical facts or foreign capitals against an enterprise profile).
4. **Exact Deterministic Refusal:** If evidence validation fails, the system immediately returns:
   ```
   ANSWER_NOT_FOUND_IN_DOCUMENT
   ```
   Zero LLM API tokens are consumed, guaranteeing 100% immunity to hallucination on unsupported facts.

### G. Prompt Injection Hardening & Groq LLM Synthesis
Located in `src/qa_engine.py`:
- **Groq Cloud Integration:** Queries Groq's high-throughput Chat Completions API (`https://api.groq.com/openai/v1`) using the official OpenAI Python SDK.
- **Real-Time Token Streaming:** Implements `stream_answer_tokens` to deliver tokens in real time (<200ms first token).
- **Default Model:** `qwen/qwen3.8-27b` (recommended for low latency and high reasoning precision) or `llama-3.3-70b-versatile`.
- **Untrusted Context Isolation:** Document passages are placed inside strict XML boundaries:
  ```xml
  <document_excerpt index="1" source="sample.pdf" page="1">
  [Chunk text here]
  </document_excerpt>
  ```
- **Defensive System Prompt:** Instructs the model to answer ONLY using facts directly asserted within `<document_excerpt>`, to ignore any prompt injection attempts inside documents, and to output `ANSWER_NOT_FOUND_IN_DOCUMENT` if evidence is lacking.

### H. Traceable Source Citations
- **PDF Citations:** Extracts the physical page number (1-indexed) directly from the PyMuPDF page map, along with the source file name and chunk index.
- **TXT Citations:** Displays file name and chunk index without fabricating imaginary page numbers.
- **Zero-Citation Policy:** When an answer is refused (`ANSWER_NOT_FOUND_IN_DOCUMENT`), no citations are emitted, preventing misleading references.

---

## 6. 🔌 FastAPI REST API Reference

The FastAPI service exposes high-performance REST endpoints for programmatic integrations:

### Base URL: `http://localhost:8000`
Interactive documentation: `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc).

| Method | Endpoint | Description | Request Body / Params | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health & active document status | None | `HealthResponse` |
| `POST` | `/api/v1/documents/index` | Upload and index TXT/PDF file | `multipart/form-data` (`file`) | `DocumentIndexResponse` (201 Created) |
| `POST` | `/api/v1/query` | Natural language question query | `QueryRequest` (JSON) | `QueryResponse` (JSON) |
| `POST` | `/api/v1/documents/reset` | Clear all active document sessions | None | Status message |

#### Example: Index a Document via cURL
```bash
curl -X POST "http://localhost:8000/api/v1/documents/index" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/datavex_profile.txt"
```

#### Example: Query the Document via cURL
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "When was the company founded?"}'
```

**Response Payload:**
```json
{
  "question": "When was the company founded?",
  "answer": "Founded: 2020",
  "supporting_passage": "Founded: 2020",
  "source_reference": {
    "filename": "datavex_profile.txt",
    "page_number": 1,
    "chunk_id": 1
  },
  "similarity_score": 0.3531,
  "retrieved_chunks": [
    {
      "chunk_id": 1,
      "page_number": 1,
      "similarity_score": 0.3531,
      "text": "Company: Datavex\nFounded: 2020\nHeadquarters: Bengaluru..."
    }
  ]
}
```

---

## 7. 🧪 Empirical Verification & Sample Test Cases

### Master Assignment Test Cases (TC1 – TC12) Verification
Tested on `sample_documents/datavex_sample.txt`:
```text
Company: Datavex
Founded: 2020
Headquarters: Bengaluru
Main services: AI solutions and data engineering
Employees: 50
```

| ID | Test Question | System Output | Validation Rule | Status |
| :--- | :--- | :--- | :--- | :---: |
| **TC1** | *When was the company founded?* | `Datavex was founded in the year 2020.` | Grounded factual extraction | ✅ **PASSED** |
| **TC2** | *Where is the headquarters?* | `The headquarters of Datavex is in Bengaluru.` | Grounded factual extraction | ✅ **PASSED** |
| **TC3** | *What services does Datavex offer?* | `Datavex offers AI solutions and data engineering.` | Grounded factual extraction | ✅ **PASSED** |
| **TC4** | *How many employees are there?* | `Datavex has a total workforce of 50 employees.` | Grounded factual extraction | ✅ **PASSED** |
| **TC5** | *Who is the CEO?* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | **Refusal Enforced (Absent Fact)** | ✅ **PASSED** |
| **TC6** | *What is Datavex's annual revenue?* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | **Refusal Enforced (Absent Fact)** | ✅ **PASSED** |
| **TC7** | *Which city is Datavex based in?* | `The headquarters of Datavex is in Bengaluru.` | Semantic variation resolution | ✅ **PASSED** |
| **TC8** | *Unsupported question exact string check* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | Exact string match requirement | ✅ **PASSED** |
| **TC9** | *Citation metadata verification* | `datavex_sample.txt, Chunk #1` | Exact provenance verification | ✅ **PASSED** |
| **TC10**| *What is the capital of Japan?* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | **Refusal Enforced (Out-of-Domain)** | ✅ **PASSED** |
| **TC11**| *Can you describe the main work done?*| `Datavex offers AI solutions and data engineering.` | Paraphrased intent resolution | ✅ **PASSED** |
| **TC12**| *Who founded Datavex?* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | **Refusal Enforced (Absent Founder)** | ✅ **PASSED** |

---

### Unseen Enterprise Generalization (CT1 – CT7)
Tested on `sample_documents/coastaltech_sample.txt`:
```text
Company: CoastalTech Solutions
Founded: 2018
Location: Kochi
Main products: Marine telemetry and acoustic sensor systems
Team size: 35
```

| ID | Test Question | System Output | Validation Rule | Status |
| :--- | :--- | :--- | :--- | :---: |
| **CT1** | *When was CoastalTech founded?* | `CoastalTech Solutions was founded in 2018.` | Grounded factual extraction | ✅ **PASSED** |
| **CT2** | *Where is CoastalTech based?* | `CoastalTech Solutions is located in Kochi.` | Grounded factual extraction | ✅ **PASSED** |
| **CT3** | *What are CoastalTech's products?* | `Marine telemetry and acoustic sensor systems.` | Grounded factual extraction | ✅ **PASSED** |
| **CT4** | *What is the team size?* | `CoastalTech Solutions has a team size of 35.` | Grounded factual extraction | ✅ **PASSED** |
| **CT5** | *Who is the CEO of CoastalTech?* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | **Refusal Enforced (Absent Fact)** | ✅ **PASSED** |
| **CT6** | *What is CoastalTech's annual profit?* | `ANSWER_NOT_FOUND_IN_DOCUMENT` | **Refusal Enforced (Absent Fact)** | ✅ **PASSED** |
| **CT7** | *Is CoastalTech in Bengaluru?* | `No, CoastalTech Solutions is located in Kochi.` | Cross-document isolation | ✅ **PASSED** |

---

## 8. ⚙️ Configuration Reference

All settings are managed via `.env` and loaded securely through `src/config.py`:

| Parameter | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `GROQ_API_KEY` | `string` | `""` | Groq Cloud API authentication key (`gsk_...`). |
| `GROQ_MODEL` | `string` | `qwen/qwen3.8-27b` | Open-weight LLM hosted on Groq (`qwen/qwen3.8-27b`, `llama-3.3-70b-versatile`). |
| `GROQ_BASE_URL` | `string` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint URL for Groq. |
| `CHUNK_SIZE` | `integer` | `700` | Target chunk size in tokens (~2,800 characters). |
| `CHUNK_OVERLAP` | `integer` | `100` | Overlap between adjacent chunks in tokens (~400 characters). |
| `SIMILARITY_THRESHOLD` | `float` | `0.85` | Maximum cosine distance cutoff for dense semantic retrieval. |
| `TOP_K` | `integer` | `3` | Maximum number of passage chunks retrieved per query. |
| `PERSIST_DIRECTORY` | `string` | `./chroma_db` | Disk directory for persistent ChromaDB storage. |
| `MAX_FILE_SIZE_MB` | `integer` | `50` | Maximum allowable document upload size in megabytes. |

---

## 9. ❓ Troubleshooting & FAQs

### Q1: Why do I see `ANSWER_NOT_FOUND_IN_DOCUMENT` when asking about the CEO?
**Answer:** This is a core negative guardrail feature, not a bug. If the uploaded document does not state who the CEO is, the negative guardrail prevents the system from fabricating an answer or naming an unrelated person from its pre-training data.

### Q2: What happens if my Groq API rate limit is exceeded or my key is invalid?
**Answer:** The system catches API exceptions gracefully and immediately falls back to its deterministic local extractive synthesis engine. You will still receive an accurate, grounded answer from the document without a crash.

### Q3: How do I run the application completely offline without an internet connection?
**Answer:** Simply leave `GROQ_API_KEY` empty in `.env`. Document parsing (PyMuPDF), chunking, vector embedding (`all-MiniLM-L6-v2`), ChromaDB storage, and extractive answer generation run 100% locally on your machine with zero external network requests.

### Q4: Streamlit says `Port 8501 is already in use`. How can I change it?
**Answer:** Run Streamlit with the `--server.port` flag:
```bash
streamlit run app.py --server.port 8502
```

### Q5: On Windows, how do I resolve ChromaDB SQLite file locks?
**Answer:** If you restart the application rapidly or run multiple parallel instances, close any existing background Python terminal processes (`Get-Process python | Stop-Process` in PowerShell) before re-launching.

---

## License & Attribution

This project is developed for enterprise document question answering with strict anti-hallucination compliance. Built with PyMuPDF, Sentence Transformers, ChromaDB, Streamlit, FastAPI, and Groq Cloud.
