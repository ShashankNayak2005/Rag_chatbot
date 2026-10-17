"""
answer_generator.py
Answer generation module communicating with the OpenAI-compatible Groq API
and zero-cost local extractive fallback synthesis.
"""

from src.qa_engine import (
    QAEngine,
    QAResponse,
    ANSWER_NOT_FOUND_IN_DOCUMENT,
    SYSTEM_INSTRUCTION,
)

__all__ = [
    "QAEngine",
    "QAResponse",
    "ANSWER_NOT_FOUND_IN_DOCUMENT",
    "SYSTEM_INSTRUCTION",
]
