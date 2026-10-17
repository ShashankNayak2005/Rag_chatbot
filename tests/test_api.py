import io
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["active_document_id"] is None


def test_upload_unsupported_file_extension(client: TestClient):
    files = {"file": ("test.docx", b"dummy word content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    response = client.post("/api/v1/documents/index", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file(client: TestClient):
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/api/v1/documents/index", files=files)
    assert response.status_code == 400
    assert "is empty" in response.json()["detail"]


def test_upload_corrupt_pdf(client: TestClient, corrupt_pdf_bytes: bytes):
    files = {"file": ("bad.pdf", corrupt_pdf_bytes, "application/pdf")}
    response = client.post("/api/v1/documents/index", files=files)
    assert response.status_code == 400
    assert "Failed to open/parse PDF" in response.json()["detail"]


def test_upload_scanned_image_only_pdf(client: TestClient, image_only_pdf_bytes: bytes):
    files = {"file": ("scanned.pdf", image_only_pdf_bytes, "application/pdf")}
    response = client.post("/api/v1/documents/index", files=files)
    assert response.status_code == 400
    assert "contains no extractable text" in response.json()["detail"]


def test_query_without_indexing_document(client: TestClient):
    response = client.post("/api/v1/query", json={"question": "What is Apollo 11?"})
    assert response.status_code == 400
    assert "No document has been indexed yet" in response.json()["detail"]


def test_query_blank_question_validation(client: TestClient, sample_txt_bytes: bytes):
    # First index a document
    files = {"file": ("space.txt", sample_txt_bytes, "text/plain")}
    index_res = client.post("/api/v1/documents/index", files=files)
    assert index_res.status_code == 201

    # Blank question
    response = client.post("/api/v1/query", json={"question": "   "})
    assert response.status_code == 422


def test_e2e_txt_indexing_and_querying(client: TestClient, sample_txt_bytes: bytes):
    # 1. Index TXT
    files = {"file": ("space.txt", sample_txt_bytes, "text/plain")}
    index_res = client.post("/api/v1/documents/index", files=files)
    assert index_res.status_code == 201
    doc_info = index_res.json()
    assert doc_info["status"] == "indexed"
    assert doc_info["filename"] == "space.txt"
    assert doc_info["total_chunks"] >= 1

    # 2. Query factual question
    query_res = client.post("/api/v1/query", json={"question": "Who landed on the Moon during Apollo 11?"})
    assert query_res.status_code == 200
    data = query_res.json()
    assert data["answer"] != "ANSWER_NOT_FOUND_IN_DOCUMENT"
    assert "Neil Armstrong" in data["answer"] or "Buzz Aldrin" in data["answer"]
    assert data["supporting_passage"] is not None
    assert data["source_reference"]["filename"] == "space.txt"
    assert data["source_reference"]["page_number"] == 1
    assert data["similarity_score"] > 0.15

    # 3. Query out-of-domain question
    unrelated_res = client.post("/api/v1/query", json={"question": "What is the capital of France?"})
    assert unrelated_res.status_code == 200
    unrelated_data = unrelated_res.json()
    assert unrelated_data["answer"] == "ANSWER_NOT_FOUND_IN_DOCUMENT"
    assert unrelated_data["supporting_passage"] is None
    assert unrelated_data["source_reference"] is None


def test_e2e_pdf_indexing_and_querying(client: TestClient, sample_pdf_bytes: bytes):
    # 1. Index PDF
    files = {"file": ("frameworks.pdf", sample_pdf_bytes, "application/pdf")}
    index_res = client.post("/api/v1/documents/index", files=files)
    assert index_res.status_code == 201
    doc_info = index_res.json()
    assert doc_info["file_type"] == "pdf"
    assert doc_info["total_pages"] == 2

    # 2. Query page 2 topic (Streamlit)
    query_res = client.post("/api/v1/query", json={"question": "How does Streamlit help data science teams?"})
    assert query_res.status_code == 200
    data = query_res.json()
    assert data["answer"] != "ANSWER_NOT_FOUND_IN_DOCUMENT"
    assert data["source_reference"]["page_number"] == 2
    assert "Streamlit" in data["answer"]


def test_when_was_company_founded_e2e(client: TestClient):
    """E2E test verifying 'When was the company founded?' for both positive and negative cases."""
    # Case A: Negative - Space document does not have company founding information
    space_doc = b"Apollo 11 was the American spaceflight that landed Neil Armstrong and Buzz Aldrin on the Moon in 1969."
    files = {"file": ("space_brief.txt", space_doc, "text/plain")}
    res = client.post("/api/v1/documents/index", files=files)
    assert res.status_code == 201

    negative_res = client.post("/api/v1/query", json={"question": "When was the company founded?"})
    assert negative_res.status_code == 200
    neg_data = negative_res.json()
    assert neg_data["answer"] == "ANSWER_NOT_FOUND_IN_DOCUMENT"
    assert neg_data["supporting_passage"] is None

    # Case B: Positive - Document contains company founding information
    company_doc = (
        b"Nexa Corp was founded on March 15, 2014, by Dr. Elena Vance and Marcus Thorne in Austin, Texas. "
        b"The enterprise specializes in real-time data pipelines."
    )
    files_company = {"file": ("nexa.txt", company_doc, "text/plain")}
    res_company = client.post("/api/v1/documents/index", files=files_company)
    assert res_company.status_code == 201

    positive_res = client.post("/api/v1/query", json={"question": "When was the company founded?"})
    assert positive_res.status_code == 200
    pos_data = positive_res.json()
    assert pos_data["answer"] != "ANSWER_NOT_FOUND_IN_DOCUMENT"
    assert "March 15, 2014" in pos_data["answer"]
    assert pos_data["supporting_passage"] is not None
    assert pos_data["source_reference"]["filename"] == "nexa.txt"
    assert pos_data["similarity_score"] >= 0.15


def test_datavex_assignment_test_cases_api_e2e(client: TestClient):
    """E2E API test running candidate test cases TC1 through TC7 on Datavex document."""
    datavex_doc = (
        b"Company: Datavex\n"
        b"Founded: 2020\n"
        b"Headquarters: Bengaluru\n"
        b"Main services: AI solutions and data engineering\n"
        b"Employees: 50"
    )
    # Upload and index document
    files = {"file": ("datavex.txt", datavex_doc, "text/plain")}
    res = client.post("/api/v1/documents/index", files=files)
    assert res.status_code == 201
    assert res.json()["status"] == "indexed"

    # TC1: When was the company founded? -> Answer 2020
    tc1_res = client.post("/api/v1/query", json={"question": "When was the company founded?"})
    assert tc1_res.status_code == 200
    assert "2020" in tc1_res.json()["answer"]
    assert tc1_res.json()["answer"] == "Founded: 2020"

    # TC2: Where is the headquarters? -> Answer Bengaluru
    tc2_res = client.post("/api/v1/query", json={"question": "Where is the headquarters?"})
    assert tc2_res.status_code == 200
    assert "Bengaluru" in tc2_res.json()["answer"]
    assert tc2_res.json()["answer"] == "Headquarters: Bengaluru"

    # TC3: How many employees are there? -> Answer 50
    tc3_res = client.post("/api/v1/query", json={"question": "How many employees are there?"})
    assert tc3_res.status_code == 200
    assert "50" in tc3_res.json()["answer"]
    assert tc3_res.json()["answer"] == "Employees: 50"

    # TC4: Who is the CEO? -> Return ANSWER_NOT_FOUND_IN_DOCUMENT
    tc4_res = client.post("/api/v1/query", json={"question": "Who is the CEO?"})
    assert tc4_res.status_code == 200
    assert tc4_res.json()["answer"] == "ANSWER_NOT_FOUND_IN_DOCUMENT"
    assert tc4_res.json()["supporting_passage"] is None

    # TC5: Empty document validation via API
    empty_files = {"file": ("empty.txt", b"", "text/plain")}
    tc5_res = client.post("/api/v1/documents/index", files=empty_files)
    assert tc5_res.status_code == 400
    assert "is empty" in tc5_res.json()["detail"]

    # TC6: Unrelated question -> Do not invent an answer
    tc6_res = client.post("/api/v1/query", json={"question": "What is the recipe for baking chocolate cookies?"})
    assert tc6_res.status_code == 200
    assert tc6_res.json()["answer"] == "ANSWER_NOT_FOUND_IN_DOCUMENT"

    # TC7: Differently worded queries
    diff1 = client.post("/api/v1/query", json={"question": "In what year did the company start?"})
    assert diff1.status_code == 200
    assert "2020" in diff1.json()["answer"]

    diff2 = client.post("/api/v1/query", json={"question": "Which city is the main office in?"})
    assert diff2.status_code == 200
    assert "Bengaluru" in diff2.json()["answer"]

    diff3 = client.post("/api/v1/query", json={"question": "What is the total staff headcount?"})
    assert diff3.status_code == 200
    assert "50" in diff3.json()["answer"]


