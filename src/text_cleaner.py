"""
text_cleaner.py
Text cleaning and normalization utilities for RAG document processing.
Preserves paragraph and sentence boundaries while normalizing unusual whitespace.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text without destroying meaningful punctuation or line breaks.
    - Normalizes unicode (NFKC).
    - Removes non-printable/control characters (except tab and newlines).
    - Normalizes Windows (CRLF) and legacy Mac (CR) line endings to UNIX (LF).
    - Replaces sequences of more than two consecutive newlines with two newlines.
    - Strips trailing whitespace per line.
    """
    if not text:
        return ""

    # Normalize unicode to standard composite form
    normalized = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unprintable/null characters
    normalized = "".join(ch for ch in normalized if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))

    # Strip excessive horizontal whitespace within lines while preserving newlines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines)

    # Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
