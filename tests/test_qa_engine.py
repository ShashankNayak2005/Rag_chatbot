"""
tests/test_qa_engine.py
Unit tests for qa_engine module: evidence extraction, citations, and hallucination guardrails.
"""

import pytest
from src.text_chunker import PassageChunk
from src.retriever import TFIDFRetriever
from src.qa_engine import QAEngine, ANSWER_NOT_FOUND_IN_DOCUMENT


@pytest.fixture
def qa_engine_fixture():
    chunks = [
        PassageChunk(
            chunk_id=1,
            source_filename="aerospace.txt",
            page_number=1,
            text=(
                "The Apollo 11 mission was commanded by Neil Armstrong and piloted by Buzz Aldrin and Michael Collins. "
                "Neil Armstrong and Buzz Aldrin landed on the lunar surface on July 20, 1969."
            ),
            char_start=0,
            char_end=202,
        ),
        PassageChunk(
            chunk_id=2,
            source_filename="aerospace.txt",
            page_number=2,
            text="The Saturn V rocket stood 363 feet tall and generated 7.5 million pounds of thrust at liftoff.",
            char_start=0,
            char_end=96,
        ),
    ]
    retriever = TFIDFRetriever(chunks)
    return QAEngine(retriever, min_similarity_threshold=0.15)


def test_factual_answer_extraction(qa_engine_fixture: QAEngine):
    res = qa_engine_fixture.answer_question("Who commanded the Apollo 11 mission?")
    assert res.is_found is True
    assert "Neil Armstrong" in res.answer
    assert res.supporting_passage is not None
    assert "Apollo 11" in res.supporting_passage
    assert res.source_filename == "aerospace.txt"
    assert res.page_number == 1
    assert res.chunk_id == 1
    assert res.similarity_score > 0.15


def test_height_query(qa_engine_fixture: QAEngine):
    res = qa_engine_fixture.answer_question("How tall was the Saturn V rocket?")
    assert res.is_found is True
    assert "363 feet" in res.answer
    assert res.page_number == 2


def test_unsupported_query_returns_exact_not_found_string(qa_engine_fixture: QAEngine):
    res = qa_engine_fixture.answer_question("What is the speed of sound in seawater?")
    assert res.is_found is False
    assert res.answer == ANSWER_NOT_FOUND_IN_DOCUMENT
    assert res.supporting_passage is None
    assert res.source_filename is None


def test_blank_question_raises_value_error(qa_engine_fixture: QAEngine):
    with pytest.raises(ValueError, match="cannot be blank"):
        qa_engine_fixture.answer_question("  ")
