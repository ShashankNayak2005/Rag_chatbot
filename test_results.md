# Automated Test Execution & Evaluation Benchmark Results

**Execution Timestamp:** 2026-10-18  
**Platform:** Windows 11 (win32)  
**Python Runtime:** Python 3.11.9  
**Core Frameworks:** pytest 9.1.1, ChromaDB 1.5.9, Sentence Transformers 6.1.0, PyMuPDF 1.23.26, OpenAI SDK 3.31.0  
**Total Automated Tests:** 125 passed, 0 failed (37.18s — optimized down from 200.60s, **5.4x speedup**)  
**Test Suite Pass Rate:** **100.0%**  

---

## 1. Summary of Test Suites

| Test Suite File | Focus Area | Total Tests | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_end_to_end.py` | Complete pipeline E2E: Datavex (TC1–TC12) & CoastalTech unseen generalization | 2 | 2 | 0 |
| `tests/test_retrieval.py` | Semantic retrieval, cosine distance ranking, document isolation, TF-IDF | 3 | 3 | 0 |
| `tests/test_answer_generation.py` | Grounded answer generation, exact fallback enforcement, prompt injection defense | 5 | 5 | 0 |
| `tests/test_citations.py` | Citation provenance, PDF page tracking, TXT chunk tracking, no duplicate text citations | 5 | 5 | 0 |
| `tests/test_error_handling.py` | Document ingestion errors, query validation, API errors (auth, 429 rate limit, timeout) | 9 | 9 | 0 |
| `tests/test_acceptance.py` | Baseline acceptance criteria (TC1 – TC7), edge cases, false-positive guardrails | 13 | 13 | 0 |
| `tests/test_groq_rag.py` | Backend RAG pipeline, mocked OpenAI SDK responses, ChromaDB namespace isolation | 17 | 17 | 0 |
| `tests/test_document_loader.py` | TXT/PDF extraction, empty files, unreadable/corrupted files, scanned PDFs | 10 | 10 | 0 |
| `tests/test_text_chunker.py` | Token-approximate chunking, overlap, natural boundary lookback, page preservation | 4 | 4 | 0 |
| `tests/test_chunker.py` | Sliding window character indices and page number metadata | 3 | 3 | 0 |
| `tests/test_retriever.py` | Semantic & TF-IDF vectorization, cosine similarity, threshold cutoff, query normalization | 5 | 5 | 0 |
| `tests/test_qa_engine.py` | Answer synthesis, evidence validation, citations, provenance, blank-question validation | 4 | 4 | 0 |
| `tests/test_rag_engine.py` | Engine initialization, thresholding, factual matching, exact string negative guard | 8 | 8 | 0 |
| `tests/test_regression_suite.py` | Tokenizer limit, negative abstention, citations, OCR detection, document isolation, API error handling | 18 | 18 | 0 |
| `tests/test_extractor.py` | Byte parsing, whitespace files, PyMuPDF page iteration | 8 | 8 | 0 |
| `tests/test_api.py` | REST integration, document upload, query endpoints, E2E | 11 | 11 | 0 |
| **TOTAL** | | **125** | **125** | **0** |

---

## 2. Master Assignment Test Cases (TC1 – TC12) Verification

**Document Source:** `sample_documents/datavex_sample.txt`
```text
Company: Datavex
Founded: 2020
Headquarters: Bengaluru
Main services: AI solutions and data engineering
Employees: 50
```

| Case | Question / Input | Expected Result | Actual Result | Distance | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **TC1** | When was the company founded? | Answer includes 2020 with valid evidence | `Datavex was founded in the year 2020.` | 0.6469 | **PASSED** |
| **TC2** | Where is the headquarters? | Answer includes Bengaluru with valid evidence | `The headquarters of Datavex is in Bengaluru.` | 0.6059 | **PASSED** |
| **TC3** | How many employees are there? | Answer includes 50 with valid evidence | `Datavex has a total workforce of 50 employees.` | 0.6579 | **PASSED** |
| **TC4** | Who is the CEO? | Exact fallback: `ANSWER_NOT_FOUND_IN_DOCUMENT` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | 0.6808 | **PASSED** |
| **TC5** | Empty or whitespace-only document | Clear validation error; no crash | `EmptyFileError: The document 'empty.txt' is empty (0 bytes).` | N/A | **PASSED** |
| **TC6** | What is the capital of Japan? | Exact fallback: `ANSWER_NOT_FOUND_IN_DOCUMENT` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | 1.0000 | **PASSED** |
| **TC7** | In what year did Datavex begin operations? | Answer includes 2020 with valid evidence | `Datavex was founded in the year 2020.` | 0.3284 | **PASSED** |
| **TC8** | Which city is Datavex based in? | Answer includes Bengaluru with valid evidence | `The headquarters of Datavex is in Bengaluru.` | 0.2861 | **PASSED** |
| **TC9** | How large is the company's workforce? | Answer includes 50 with valid evidence | `Datavex has a total workforce of 50 employees.` | 0.6850 | **PASSED** |
| **TC10** | What kind of services does Datavex offer? | Answer includes AI solutions and data engineering | `Datavex offers AI solutions and data engineering.` | 0.3743 | **PASSED** |
| **TC11** | Who established Datavex? | Exact fallback: `ANSWER_NOT_FOUND_IN_DOCUMENT` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | 0.3039 | **PASSED** |
| **TC12** | What is Datavex's annual revenue? | Exact fallback: `ANSWER_NOT_FOUND_IN_DOCUMENT` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | 0.4320 | **PASSED** |

---

## 3. Unseen Data Generalization (CoastalTech CT1 – CT7)

**Document Source:** `sample_documents/coastaltech_sample.txt`
```text
CoastalTech was established in 2018. Its main office is located in Mangaluru.
The organization has a workforce of 120 people. Its services include cloud consulting
and cybersecurity.
```

| Case | Question / Input | Expected Result | Actual Result | Distance | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **CT1** | When was CoastalTech established? | Answer includes 2018 with valid evidence | `CoastalTech was established in 2018.` | 0.2529 | **PASSED** |
| **CT2** | Where is its main office? | Answer includes Mangaluru with valid evidence | `Its main office is located in Mangaluru.` | 0.5833 | **PASSED** |
| **CT3** | How many people work for the organization? | Answer includes 120 with valid evidence | `The organization has a workforce of 120 people.` | 0.7020 | **PASSED** |
| **CT4** | What areas does it provide services in? | Answer includes cloud consulting and cybersecurity | `Its services include cloud consulting and cybersecurity.` | 0.6170 | **PASSED** |
| **CT5** | Which services does CoastalTech offer? | Answer includes cloud consulting and cybersecurity | `Its services include cloud consulting and cybersecurity.` | 0.2974 | **PASSED** |
| **CT6** | What is CoastalTech's annual revenue? | Exact fallback: `ANSWER_NOT_FOUND_IN_DOCUMENT` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | 0.3900 | **PASSED** |
| **CT7** | Who is the CEO of CoastalTech? | Exact fallback: `ANSWER_NOT_FOUND_IN_DOCUMENT` | `ANSWER_NOT_FOUND_IN_DOCUMENT` | 0.3935 | **PASSED** |

---

## 4. Automated Benchmark Evaluation Report (`evaluation/run_evaluation.py`)

**Execution Command:** `python evaluation/run_evaluation.py`  
**Dataset:** `evaluation/evaluation_dataset.json` (22 multi-document test cases across Datavex, CoastalTech, and Solar PV Report)

| Metric | Target | Actual Measured Score | Denominator / Samples |
| :--- | :---: | :---: | :---: |
| **Supported-Answer Accuracy** | > 90% | **100.00%** | 14 / 14 questions |
| **Unsupported-Question Rejection Rate** | 100% | **100.00%** | 8 / 8 absent facts |
| **Retrieval Hit Rate** | > 85% | **92.86%** | 13 / 14 supported facts |
| **Paraphrase Success Rate** | > 85% | **100.00%** | 4 / 4 paraphrased queries |
| **Citation Validity Rate** | 100% | **100.00%** | 14 / 14 generated citations |
| **Unsupported-Claim Rate** | 0% | **0.00%** | 0 / 8 hallucinations |
| **Document Validation Success Rate** | 100% | **100.00%** | 4 / 4 edge cases (empty, whitespace, corrupt, unsupported) |
| **Operational Error Handling Rate** | 100% | **100.00%** | 4 / 4 failure simulations (auth, 429 rate limit, timeout, missing key) |

---

## 5. Security & Isolation Verification

- **Cross-Document Contamination Defense:** Tested in `test_document_isolation_prevents_cross_contamination` and `test_document_isolation_in_retrieval`. Queries against Document A strictly filter by `document_id`, preventing evidence from Document B from leaking.
- **Prompt Injection Defense:** Tested in `test_prompt_injection_safety` and `test_prompt_injection_defense`. Adversarial injection instructions (`"Ignore all previous instructions..."`) embedded inside uploaded document excerpts are encapsulated within `<document_excerpt>` untrusted data tags and disregarded.
- **Provider Neutrality:** The UI layer (`app.py`, `frontend/app.py`) contains zero provider-specific branding, model names, or API endpoint disclosures.
