import io
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.state_manager import state_manager


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure state manager is reset before and after each test."""
    state_manager.clear()
    yield
    state_manager.clear()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def sample_txt_content() -> str:
    return (
        "The Apollo program was conceived during Dwight D. Eisenhower's administration. "
        "It was subsequently dedicated to John F. Kennedy's national goal of landing a man on the Moon "
        "and returning him safely to the Earth by the end of the 1960s.\n\n"
        "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
        "Commander Neil Armstrong and Lunar Module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle "
        "on July 20, 1969. Armstrong became the first person to step onto the lunar surface six hours later.\n\n"
        "Project Gemini was the second crewed spaceflight program of NASA, bridging Mercury and Apollo."
    )


@pytest.fixture
def sample_txt_bytes(sample_txt_content: str) -> bytes:
    return sample_txt_content.encode("utf-8")


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a valid two-page PDF with clear extractable text."""
    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 100),
        "FastAPI is a modern, fast web framework for building APIs with Python 3.8+.\n"
        "It is based on standard Python type hints and provides automatic interactive documentation.\n"
        "FastAPI is built on top of Starlette and Pydantic.",
        fontsize=12,
    )

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 100),
        "Streamlit turns data scripts into shareable web apps in minutes.\n"
        "All in pure Python. No frontend experience required.\n"
        "It enables rapid prototyping for machine learning and data science teams.",
        fontsize=12,
    )

    pdf_buffer = io.BytesIO()
    doc.save(pdf_buffer)
    doc.close()
    return pdf_buffer.getvalue()


@pytest.fixture
def image_only_pdf_bytes() -> bytes:
    """Generate a PDF page with 0 text characters (simulating a scanned document)."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # blank page without text
    pdf_buffer = io.BytesIO()
    doc.save(pdf_buffer)
    doc.close()
    return pdf_buffer.getvalue()


@pytest.fixture
def corrupt_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n%corrupted broken header binary data\xff\xfe\x00\x00\x00"
