from backend.app.services.extractor import ExtractionResult, ExtractedPage
from backend.app.services.chunker import create_chunks_from_document, split_text_into_chunks


def test_split_text_into_chunks_short_text():
    text = "Short sentence."
    chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0][2] == text


def test_split_text_into_chunks_long_text():
    text = (
        "This is sentence one. This is sentence two. "
        "This is sentence three. This is sentence four. "
        "This is sentence five. This is sentence six."
    )
    chunks = split_text_into_chunks(text, chunk_size=60, chunk_overlap=15)
    assert len(chunks) > 1
    # Check that chunks have non-empty text and preserve order
    for start, end, chunk_text in chunks:
        assert len(chunk_text) > 0
        assert chunk_text in text


def test_create_chunks_preserves_page_numbers_and_metadata():
    extraction = ExtractionResult(
        filename="multipage.pdf",
        file_type="pdf",
        pages=[
            ExtractedPage(page_number=1, text="Page 1 content about machine learning and models."),
            ExtractedPage(page_number=2, text="Page 2 content discussing retrieval augmented generation."),
        ],
        total_characters=98,
    )

    chunks = create_chunks_from_document(extraction, chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == 1
    assert chunks[0].page_number == 1
    assert chunks[0].source_filename == "multipage.pdf"
    assert "machine learning" in chunks[0].text

    assert chunks[1].chunk_id == 2
    assert chunks[1].page_number == 2
    assert chunks[1].source_filename == "multipage.pdf"
    assert "retrieval augmented generation" in chunks[1].text
