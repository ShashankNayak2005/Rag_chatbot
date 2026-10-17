# Document Q&A — Professional Document Question-Answering System Using RAG

An enterprise-grade, evidence-grounded Retrieval-Augmented Generation (RAG) system for question answering on text-based PDF and TXT documents. Built with Python, PyMuPDF, local Sentence Transformers, persistent ChromaDB, and backend LLM Chat Completions via the official OpenAI-compatible SDK.

The system answers questions strictly from the factual contents of uploaded documents, guarantees source provenance through traceable citations, and strictly returns `ANSWER_NOT_FOUND_IN_DOCUMENT` when questions are unsupported.

---

## 1. Project Overview

**Document Q&A** is a general-purpose, document question-answering application designed for enterprise document exploration, technical reports, company profiles, financial disclosures, and contracts. It operates on arbitrary uploaded documents without reliance on hardcoded answers, document-specific shortcuts, or external embedding APIs.

---

## 2. Problem Statement

Standard Large Language Models frequently suffer from hallucinations, out-of-date information, and an inability to trace assertions to source material. Conversely, naive RAG pipelines often return irrelevant chunks or summarize off-topic text when asked about absent entities (e.g., answering "Bengaluru" when asked about the CEO because the company profile contains its headquarters). 

**Document Q&A** solves these challenges through:
1. **Strict Factual Predicate Verification:** Differentiates between topical similarity and factual sufficiency.
2. **Defensive Prompt Boundaries:** Treats document excerpts as untrusted data within `<document_excerpt>` delimiters, neutralizing prompt-injection attacks.
3. **Deterministic Negative Guardrails:** Returns exactly `ANSWER_NOT_FOUND_IN_DOCUMENT` for unsupported facts.
4. **End-to-End Citation Provenance:** Provides verifiable citations with physical PDF page numbers and TXT chunk references.

---

## 3. Key Features

- **Multi-Format Ingestion:** Extracts readable text page-by-page from PDF and TXT files.
- **Robust Ingestion Validation:** Detects empty files (0 bytes), whitespace-only documents, corrupt PDFs, and scanned/image-only PDFs lacking extractable text.
- **Sentence-Aware Chunking:** Configurable chunk size (~700 tokens) and overlap (~100 tokens) with natural sentence and paragraph boundary lookback.
- **100% Local Embeddings:** High-speed, local vectorization using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) on CPU without external API calls or costs.
- **Isolated Persistent Vector Store:** ChromaDB with cosine distance metric (`hnsw:space: cosine`), isolated by content-hashed `document_id` to prevent cross-document contamination.
- **Grounded Answer Generation:** Backend natural language generation using OpenAI-compatible Chat Completions API with a zero-cost local extractive fallback.
- **Strict Evidence Validation:** Rejects unsupported facts, missing financial metrics, absent personnel, and out-of-domain questions with the exact string `ANSWER_NOT_FOUND_IN_DOCUMENT`.
- **Traceable Source Citations:** Excerpts, physical PDF page numbers, and chunk IDs attached to every grounded response.
- **Provider-Neutral Interface:** Clean, professional Streamlit interface free of vendor logos, provider names, or implementation leakage.
- **Comprehensive Testing & Evaluation:** 105 automated pytest tests passing 100%, automated evaluation benchmark reporting 8 metrics, and one-click test case chips.

---

## 4. Architecture Overview

```
[Uploaded Document: PDF / TXT]
              │
              ▼
[Document Loader (PyMuPDF / UTF-8)] ──> Validate file type, size, non-empty text, page boundaries
              │
              ▼
[Text Chunker] ─────────────────────> ~700 tokens, 100 token overlap, sentence lookback, document_id hashing
              │
              ▼
[Local Embedding Service] ──────────> all-MiniLM-L6-v2 (Local CPU, 384 dimensions)
              │
              ▼
[Persistent ChromaDB Store] ────────> hnsw:space: cosine, isolated by document_id namespace
              │
              ▼
[Semantic & Keyword Retriever] ─────> Query embedded locally; candidate passages retrieved with cosine distance
              │
              ▼
[Evidence Validation Guard] ────────> Factual predicate verification (detects absent facts, CEO, revenue)
              │
              ├── [Evidence Insufficient] ──> Returns exactly "ANSWER_NOT_FOUND_IN_DOCUMENT" (No API call)
              │
              └── [Evidence Validated]
                      │
                      ▼
              [Backend LLM Generation] ─────> OpenAI-compatible Chat Completions API (or local extractive fallback)
                      │                       Untrusted data boundary delimiters (<document_excerpt>)
                      ▼
              [Grounded Answer & Source Citations]
```

---

## 5. Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive, professional chat interface with persistent history and sidebar controls |
| **PDF Extraction** | PyMuPDF (`fitz`) | High-speed, page-by-page text extraction preserving exact physical page numbers |
| **Text Processing** | Python Standard Library | Regex, unicodedata, and sentence-aware chunking |
| **Local Embeddings** | Sentence Transformers | `all-MiniLM-L6-v2` (384-dim), running 100% locally on CPU without external API charges |
| **Vector Database** | ChromaDB | Local persistent vector database with cosine space (`hnsw:space: cosine`) |
| **Keyword Retrieval** | scikit-learn TF-IDF | Complementary lexical matching for exact keywords, abbreviations, and numerical terms |
| **Backend LLM API** | OpenAI-compatible SDK | Configured to communicate with the Groq API endpoint for natural language answer synthesis |
| **Configuration** | python-dotenv | Secure environment variable management via `.env` |
| **Testing** | pytest | 105 automated unit, acceptance, error handling, citation, and end-to-end tests |

---

## 6. Project Folder Structure

```
document-qa-rag/
├── app.py                         # Main Streamlit user interface
├── requirements.txt               # Locked and verified dependencies
├── README.md                      # Comprehensive system documentation
├── sample_output.md               # Empirical sample outputs and test run demonstrations
├── test_results.md                # Automated test execution and evaluation results
├── pytest.ini                     # Pytest configuration (targets tests/ directory)
├── run_test_cases.py              # CLI test runner for TC1-TC12 and CoastalTech
├── .env.example                   # Environment configuration template (placeholders only)
├── .gitignore                     # Git ignore rules for virtual environments, secrets, databases
├── src/                           # Core application backend modules
│   ├── __init__.py                # Package exports and module facades
│   ├── config.py                  # Centralized configuration management
│   ├── document_loader.py         # Multi-format document ingestion and validation
│   ├── text_cleaner.py            # Text normalization and whitespace sanitization
│   ├── text_chunker.py            # Sentence-aware chunking and token counting
│   ├── chunker.py                 # Chunker alias wrapper
│   ├── embedding_service.py       # Local Sentence Transformers embedding generator
│   ├── embeddings.py              # Embedding service alias wrapper
│   ├── vector_store.py            # ChromaDB persistent vector storage and metadata isolation
│   ├── retriever.py               # Semantic retriever, cosine distance filtering, TF-IDF
│   ├── evidence_validator.py      # Evidence validation facade
│   ├── answer_generator.py        # Answer generation facade
│   ├── qa_engine.py               # Core QA engine, guardrails, and LLM communication
│   └── pipeline.py                # High-level DocumentQAPipeline facade
├── tests/                         # Automated pytest test suite
│   ├── conftest.py                # Shared pytest fixtures
│   ├── test_document_loader.py    # Ingestion, validation, and error tests
│   ├── test_chunker.py            # Chunking and boundary tests
│   ├── test_text_chunker.py       # Sliding window character offset and overlap tests
│   ├── test_retrieval.py          # Semantic retrieval, distance ranking, and isolation tests
│   ├── test_retriever.py          # Vector store retrieval and threshold tests
│   ├── test_answer_generation.py  # Grounded synthesis and prompt injection tests
│   ├── test_citations.py          # Provenance tracking, PDF pages, and TXT chunk tests
│   ├── test_error_handling.py     # Ingestion errors, query errors, and API exception tests
│   ├── test_end_to_end.py         # Complete pipeline E2E tests (TC1-TC12 + CoastalTech)
│   ├── test_acceptance.py         # Baseline acceptance criteria tests (TC1-TC7)
│   ├── test_deepseek_rag.py       # Comprehensive backend RAG integration tests
│   ├── test_qa_engine.py          # Unit tests for QAEngine methods
│   ├── test_rag_engine.py         # Factual matching and negative guardrail tests
│   ├── test_extractor.py          # Byte extraction and page iteration tests
│   └── test_api.py                # REST API endpoint tests
├── evaluation/                    # Automated evaluation benchmark system
│   ├── evaluation_dataset.json    # Benchmark dataset across multiple documents
│   ├── run_evaluation.py          # Automated evaluation runner script
│   └── evaluation_report.json     # Generated evaluation benchmark metrics
├── sample_documents/              # Required and unseen test documents
│   ├── datavex_sample.txt         # Official required Datavex sample document
│   └── coastaltech_sample.txt     # Unseen synthetic CoastalTech test document
├── sample_docs/                   # Additional test assets
│   ├── sample_article.pdf         # Multi-page test PDF
│   └── sample_report.txt          # Technical report on Renewable Energy Solar PV
└── data/                          # Runtime data and sample files
    └── datavex_profile.txt        # Supplementary company profile text
```

---

## 7. How Document Extraction Works

The document extraction subsystem is implemented in `src/document_loader.py`:

- **Pre-Ingestion Validation:** Inspects file extensions and file sizes before reading content. Rejects unsupported extensions (`.exe`, `.docx`, `.csv`, etc.) and oversized uploads.
- **PDF Extraction:**
  - Opens the PDF using PyMuPDF (`fitz.open(stream=..., filetype="pdf")`).
  - Iterates page by page, extracting plain text via `page.get_text("text")`.
  - Preserves 1-indexed physical page numbers.
  - Detects password-protected PDFs and raises `UnreadableFileError`.
  - Detects scanned or image-only PDFs that contain zero extractable text and raises `NoExtractableTextError` with an actionable message explaining that image-only PDFs require OCR.
  - Detects corrupted PDF byte headers or unreadable structures and raises `UnreadableFileError`.
- **TXT Extraction:**
  - Decodes byte streams using UTF-8 with automatic fallback to Latin-1.
  - Handles UTF-8 Byte Order Marks (BOM) transparently.
  - Detects empty files (0 bytes) and raises `EmptyFileError`.
  - Normalizes carriage returns (`\r\n` $\rightarrow$ `\n`) and verifies that the file contains readable non-whitespace text. If only whitespace is present, raises `NoExtractableTextError`.
- **Document Metadata:** Computes a unique `document_id` using SHA-256 over the document content, which partitions all vector storage and retrieval.

---

## 8. How Chunking and Metadata Preservation Work

Implemented in `src/text_chunker.py`:

- **Sentence-Aware Chunking:** Operates with a target chunk size of **700 tokens** (~2,800 characters) and a target overlap of **100 tokens** (~400 characters).
- **Boundary Lookback:** When a chunk boundary is reached, the chunker scans backward within a configurable window (up to 200 characters) to split at natural sentence delimiters (`. `, `! `, `? `) or paragraph breaks (`\n\n`), preventing the mid-sentence truncation of facts.
- **Empty & Duplicate Elimination:** Strips leading/trailing whitespace, discards zero-length chunks, and deduplicates identical passages.
- **Traceable Chunk Metadata:** Each `PassageChunk` preserves:
  - `chunk_id`: Monotonically increasing chunk index (1-indexed).
  - `document_id`: Content-hashed document identifier.
  - `source_filename`: Original filename.
  - `page_number`: 1-indexed page number for PDFs; `1` for single-unit TXT documents.
  - `char_start` & `char_end`: Character offsets in the original extracted document text.

---

## 9. How Local Embeddings and Vector Retrieval Work

Implemented in `src/embedding_service.py` and `src/vector_store.py`:

- **Embedding Model:** Uses `sentence-transformers/all-MiniLM-L6-v2`, running locally on CPU.
- **Zero API Cost:** All embeddings for document chunks and user queries are generated locally. No external embedding service is called.
- **Vector Storage:** Persistent ChromaDB instance on disk (default directory: `./chroma_db`).
- **Distance Metric:** Configured with cosine distance (`{"hnsw:space": "cosine"}`). In cosine space:
  $$\text{Cosine Distance} = 1.0 - \text{Cosine Similarity}$$
  Smaller distance indicates greater semantic similarity (distance 0.0 is identical; distance 1.0 is orthogonal).
- **Partitioning & Isolation:** All chunk embeddings stored in ChromaDB include `document_id` in their metadata. When queries execute, ChromaDB applies a metadata filter (`where={"document_id": active_document_id}`), guaranteeing that chunks from previous uploads never contaminate the current query.

---

## 10. How Hybrid Retrieval Works

Implemented in `src/retriever.py`:

- **Dense Semantic Retrieval:** Queries are vectorized locally using `all-MiniLM-L6-v2` and searched against ChromaDB with a configurable distance cutoff (default: `0.85`).
- **Lexical Keyword Retrieval:** A local `TFIDFRetriever` computes term-frequency inverse-document-frequency vectors over the document chunks, providing complementary retrieval for exact acronyms, numbers, and dates.
- **Reciprocal Rank Fusion / Candidate Fallback:** If semantic distance exceeds the strict cutoff for short keyword queries or receipt key-values, candidate passages up to distance `0.98` are retrieved from the isolated document partition and validated against token overlap and key-value line patterns, eliminating false negatives while preventing hallucinations.

---

## 11. Evidence Validation and Unsupported Questions

Implemented in `src/qa_engine.py`:

Before invoking the LLM, `validate_evidence()` evaluates whether the retrieved document passages contain sufficient factual evidence to answer the question:

1. **Negative Guardrails for Absent Facts:**
   - **CEO / Leadership:** If the query asks for the CEO, founder, or president, but the retrieved passages contain no leadership entities, retrieval is blocked.
   - **Revenue / Financials:** If the query asks for annual revenue, profit, or sales, but no financial figures or currencies exist in the excerpts, retrieval is blocked.
   - **Out-of-Domain Queries:** Questions completely unrelated to the document (e.g., "What is the capital of Japan?" queried against a company profile) are rejected.
2. **Key-Value Line Pattern Alignment:** Recognizes key-value structures (e.g., `Founded: 2020`, `Headquarters: Bengaluru`, `Main services: AI solutions`).
3. **Exact Fallback String:** When evidence is insufficient, the system returns strictly:
   ```
   ANSWER_NOT_FOUND_IN_DOCUMENT
   ```
   No conversational padding, punctuation, or explanations are appended.

---

## 12. Citation and Source Evidence System

Every grounded answer is accompanied by traceable source citations:
- **PDF Documents:** Displays `Filename`, physical `Page Number`, `Chunk ID`, and the exact `Passage Excerpt`.
- **TXT Documents:** Displays `Filename`, `Chunk ID`, and the exact `Passage Excerpt` (without inventing PDF page numbers).
- **Unsupported Questions:** Zero citations are emitted when the answer is `ANSWER_NOT_FOUND_IN_DOCUMENT`.

---

## 13. Backend LLM Configuration (Groq API)

The system uses the official `openai` Python SDK configured with Groq's Chat Completions API base URL and supported model:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

### Supported Groq Models
- `qwen/qwen3.8-27b` (recommended default, fast and highly grounded)
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`

### Zero-Cost Local Extractive Fallback
If no API key is provided, or if `use_deepseek=False` is passed, the system automatically falls back to deterministic local extractive answer extraction without throwing an error or incurring API costs.

---

## 14. Installation & Setup Instructions (Windows PowerShell)

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "c:\Users\shash\Music\rag chatbot"
```

### Step 2: Create and Activate a Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` and set your `GROQ_API_KEY`:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

---

## 15. Running the Application

Launch the Streamlit web application:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 16. Executing Automated Tests

Run the full pytest suite (105 tests, 100% offline, zero API charges):
```powershell
python -m pytest -v
```

Run specific test modules:
```powershell
# Run end-to-end assignment tests (TC1-TC12 + CoastalTech)
python -m pytest tests/test_end_to_end.py -v

# Run error handling and ingestion tests
python -m pytest tests/test_error_handling.py -v

# Run citation tests
python -m pytest tests/test_citations.py -v

# Run acceptance tests
python -m pytest tests/test_acceptance.py -v
```

Run the standalone CLI test runner:
```powershell
python run_test_cases.py
```

---

## 17. Running the Evaluation Benchmark

Run the automated evaluation benchmark across all test documents:
```powershell
python evaluation/run_evaluation.py
```
The script evaluates all 22 test cases in `evaluation/evaluation_dataset.json`, measures 8 empirical metrics, and saves the results to `evaluation/evaluation_report.json`.

---

## 18. Sample Document & Example Queries

### Sample Document: Datavex (`sample_documents/datavex_sample.txt`)
```text
Company: Datavex
Founded: 2020
Headquarters: Bengaluru
Main services: AI solutions and data engineering
Employees: 50
```

### Supported Questions & Expected Outputs
| Question | Expected Output | Citation |
| :--- | :--- | :--- |
| `When was the company founded?` | `Datavex was founded in the year 2020.` | `datavex_sample.txt`, Chunk #1 |
| `Where is the headquarters?` | `The headquarters of Datavex is in Bengaluru.` | `datavex_sample.txt`, Chunk #1 |
| `How many employees are there?` | `Datavex has a total workforce of 50 employees.` | `datavex_sample.txt`, Chunk #1 |
| `What kind of services does Datavex offer?` | `Datavex offers AI solutions and data engineering.` | `datavex_sample.txt`, Chunk #1 |
| `Which city is Datavex based in?` | `The headquarters of Datavex is in Bengaluru.` | `datavex_sample.txt`, Chunk #1 |

### Unsupported Questions & Exact Fallback
| Question | Expected Output | Reason |
| :--- | :--- | :--- |
| `Who is the CEO?` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | CEO/leadership absent from document |
| `What is Datavex's annual revenue?` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | Financial figures absent from document |
| `Who established Datavex?` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | Founder absent from document |
| `What is the capital of Japan?` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | Query out of scope of document |

---

## 19. Security, Privacy & Reliability

1. **Credential Secrecy:** API keys are read strictly from `.env` via `src/config.py`. They are never passed to the frontend, never stored in ChromaDB metadata, and never logged in error messages.
2. **Prompt Injection Hardening:** Document content is treated strictly as untrusted data within `<document_excerpt>` XML tags. The system prompt instructs the model to ignore instructions contained within excerpts.
3. **Session & Document Isolation:** ChromaDB queries are filtered by `document_id`. Uploading a new document switches the active document namespace, ensuring no stale or unrelated evidence leaks into queries.
4. **Offline Safety:** The entire embedding and retrieval pipeline runs locally. Document contents are never sent to external services except for the specific retrieved excerpts sent to the configured backend Chat Completions endpoint.

---

## 20. Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `ImportError: cannot import name ...` | Package version mismatch | Run `pip install -r requirements.txt` |
| `AuthenticationError: Invalid API Key` | Missing or expired Groq key | Check `GROQ_API_KEY` in `.env` |
| `No extractable text found in PDF` | Scanned or image-only PDF | Ensure PDF contains digital text, or run OCR prior to upload |
| Streamlit port in use | Port 8501 is occupied | Run `streamlit run app.py --server.port 8502` |
| ChromaDB permission error on Windows | Locked SQLite file | Stop any other background Python processes accessing `./chroma_db` |

---

## 21. Limitations & Future Roadmap

- **OCR Support:** Scanned and image-only PDFs currently trigger a clean validation notice; integrating Tesseract OCR would enable text extraction from scanned images.
- **Table Extraction:** PyMuPDF extracts raw text; specialized table extractors (e.g., pdfplumber) could enhance extraction of complex financial grids.
- **Hybrid Reranking:** Adding a local cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) would further refine top-3 chunk ordering for complex multi-hop reasoning.
