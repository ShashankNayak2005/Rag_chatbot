"""
evidence_validator.py
Evidence validation and anti-hallucination guardrails module.
Evaluates whether retrieved passages factually support user inquiries.
"""

from src.qa_engine import (
    validate_evidence,
    split_into_sentences,
    ANSWER_NOT_FOUND_IN_DOCUMENT,
)

__all__ = [
    "validate_evidence",
    "split_into_sentences",
    "ANSWER_NOT_FOUND_IN_DOCUMENT",
]
