import pytest
from backend.app.services.chunker import DocumentChunk
from backend.app.services.rag_engine import RAGEngine, EXACT_NOT_FOUND_STRING


@pytest.fixture
def rag_engine_instance():
    chunks = [
        DocumentChunk(
            chunk_id=1,
            source_filename="space_history.txt",
            page_number=1,
            text=(
                "The Apollo program was dedicated to John F. Kennedy's national goal of landing a man on the Moon "
                "and returning him safely to the Earth by the end of the 1960s."
            ),
            char_start=0,
            char_end=154,
        ),
        DocumentChunk(
            chunk_id=2,
            source_filename="space_history.txt",
            page_number=1,
            text=(
                "Apollo 11 commander Neil Armstrong and Lunar Module pilot Buzz Aldrin landed on July 20, 1969. "
                "Neil Armstrong became the first person to step onto the lunar surface."
            ),
            char_start=155,
            char_end=333,
        ),
        DocumentChunk(
            chunk_id=3,
            source_filename="space_history.txt",
            page_number=2,
            text=(
                "Project Gemini was the second crewed spaceflight program of NASA. "
                "It conducted ten crewed missions in 1965 and 1966."
            ),
            char_start=0,
            char_end=138,
        ),
    ]
    return RAGEngine(chunks=chunks, min_similarity_threshold=0.15)


def test_rag_engine_initialization_with_empty_chunks():
    with pytest.raises(ValueError, match="empty chunks list"):
        RAGEngine(chunks=[])


def test_retrieval_returns_highest_scoring_chunk(rag_engine_instance: RAGEngine):
    results = rag_engine_instance.retrieve("Who was the commander of Apollo 11?", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == 2
    assert "Neil Armstrong" in top_chunk.text
    assert score > 0.2


def test_answer_question_factual_match(rag_engine_instance: RAGEngine):
    response = rag_engine_instance.answer_question("When did Apollo 11 land on the Moon?")
    assert response.answer != EXACT_NOT_FOUND_STRING
    assert "July 20, 1969" in response.answer or "Neil Armstrong" in response.answer
    assert response.supporting_passage is not None
    assert response.source_reference is not None
    assert response.source_reference.chunk_id == 2
    assert response.source_reference.page_number == 1
    assert response.similarity_score > 0.15


def test_unsupported_question_returns_exact_not_found_string(rag_engine_instance: RAGEngine):
    # Topic completely absent from the space history document
    response = rag_engine_instance.answer_question("What is the recipe for chocolate chip cookies?")
    assert response.answer == EXACT_NOT_FOUND_STRING
    assert response.supporting_passage is None
    assert response.source_reference is None
    assert response.similarity_score == 0.0


def test_weak_marginal_query_below_threshold_returns_not_found(rag_engine_instance: RAGEngine):
    # Query with low relevance
    response = rag_engine_instance.answer_question(
        "Is there any deep ocean submarine exploration discussed here?",
        threshold=0.30,
    )
    assert response.answer == EXACT_NOT_FOUND_STRING
    assert response.supporting_passage is None
    assert response.source_reference is None


def test_query_when_was_company_founded_negative_case(rag_engine_instance: RAGEngine):
    """Test case: 'When was the company founded?' against a document with no company info."""
    response = rag_engine_instance.answer_question("When was the company founded?")
    assert response.answer == EXACT_NOT_FOUND_STRING
    assert response.supporting_passage is None
    assert response.source_reference is None
    assert response.similarity_score == 0.0


def test_query_when_was_company_founded_positive_case():
    """Test case: 'When was the company founded?' against a document that contains company founding info."""
    chunks = [
        DocumentChunk(
            chunk_id=1,
            source_filename="company_profile.txt",
            page_number=1,
            text=(
                "Nexa Corp was founded on March 15, 2014, by Dr. Elena Vance and Marcus Thorne in Austin, Texas. "
                "The company was established with the vision of developing enterprise cloud intelligence platforms."
            ),
            char_start=0,
            char_end=202,
        ),
        DocumentChunk(
            chunk_id=2,
            source_filename="company_profile.txt",
            page_number=1,
            text="The company specializes in automated data pipelines and real-time streaming architectures.",
            char_start=203,
            char_end=295,
        ),
    ]
    engine = RAGEngine(chunks=chunks, min_similarity_threshold=0.15)
    response = engine.answer_question("When was the company founded?")

    assert response.answer != EXACT_NOT_FOUND_STRING
    assert "March 15, 2014" in response.answer
    assert response.supporting_passage is not None
    assert "Nexa Corp was founded on March 15, 2014" in response.supporting_passage
    assert response.source_reference is not None
    assert response.source_reference.chunk_id == 1
    assert response.source_reference.page_number == 1
    assert response.similarity_score >= 0.15


def test_user_datavex_key_value_test_case():
    """Verify exact specific line extraction on user's Datavex test case."""
    text = (
        "Company: Datavex\n"
        "Founded: 2020\n"
        "Headquarters: Bengaluru\n"
        "Main services: AI solutions and data engineering\n"
        "Employees: 50"
    )
    chunk = DocumentChunk(
        chunk_id=1,
        source_filename="datavex.txt",
        page_number=1,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    engine = RAGEngine(chunks=[chunk], min_similarity_threshold=0.15)

    # TC1: When was the company founded? -> Answer 2020
    res_founded = engine.answer_question("When was the company founded?")
    assert "2020" in res_founded.answer
    assert res_founded.answer == "Founded: 2020"
    assert res_founded.supporting_passage == text

    # TC2: Where is the headquarters? -> Answer Bengaluru
    res_hq = engine.answer_question("Where is the headquarters?")
    assert "Bengaluru" in res_hq.answer
    assert res_hq.answer == "Headquarters: Bengaluru"

    # TC3: How many employees are there? -> Answer 50
    res_emp = engine.answer_question("How many employees are there?")
    assert "50" in res_emp.answer
    assert res_emp.answer == "Employees: 50"

    # TC4: Who is the CEO? -> Return ANSWER_NOT_FOUND_IN_DOCUMENT
    res_ceo = engine.answer_question("Who is the CEO?")
    assert res_ceo.answer == EXACT_NOT_FOUND_STRING
    assert res_ceo.supporting_passage is None

    # TC6: Unrelated question -> Do not invent an answer
    res_unrelated = engine.answer_question("What is the speed of light in vacuum?")
    assert res_unrelated.answer == EXACT_NOT_FOUND_STRING
    assert res_unrelated.supporting_passage is None

    # TC7: Questions worded differently from the document
    res_diff_founding = engine.answer_question("In what year did the company start?")
    assert "2020" in res_diff_founding.answer
    assert res_diff_founding.answer == "Founded: 2020"

    res_diff_hq = engine.answer_question("Which city is the main office in?")
    assert "Bengaluru" in res_diff_hq.answer
    assert res_diff_hq.answer == "Headquarters: Bengaluru"

    res_diff_staff = engine.answer_question("What is the total staff headcount?")
    assert "50" in res_diff_staff.answer
    assert res_diff_staff.answer == "Employees: 50"

    res_diff_services = engine.answer_question("What services does Datavex provide?")
    assert "AI solutions" in res_diff_services.answer
    assert "data engineering" in res_diff_services.answer


