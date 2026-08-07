"""PDF text extraction and clause splitting for the Contract Risk Detector API.

This module provides:
- extract_text_from_pdf: Extracts raw text from PDF bytes using pdfplumber.
- split_into_clauses: Heuristic clause splitter using regex patterns.
"""

import io
import re
from typing import List

import pdfplumber

from src.utils import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file provided as raw bytes.

    Uses pdfplumber to read each page and join the extracted text
    with newlines.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the PDF contains zero extractable text (e.g.
            scanned image PDFs without OCR). This module does NOT
            attempt OCR — if the PDF is image-only, it will fail
            clearly with this error.
    """
    text_parts: List[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(
            "No extractable text found in the PDF. "
            "This may be a scanned/image-only PDF. OCR is not supported."
        )

    logger.info("Extracted %d characters from PDF (%d pages with text)",
                len(full_text), len(text_parts))
    return full_text


def split_into_clauses(text: str) -> List[str]:
    """Split contract text into individual clauses using heuristics.

    IMPORTANT: This is a heuristic regex-based splitter, NOT a
    legal-grade clause parser. It may not align with real clause
    boundaries in all contracts. It uses two strategies:

    1. Primary: Split on numbered section patterns (e.g. "1.", "2.3",
       "(a)", etc.) at the start of a line.
    2. Fallback: If no numbered sections are found, split on
       double-newline paragraph breaks.

    Clauses shorter than 20 characters are filtered out as likely
    headers or noise.

    Args:
        text: The full contract text to split.

    Returns:
        A list of clause text strings.
    """
    # Strategy 1: Split on numbered section headings at line start
    # Matches patterns like: "1. ", "2.3 ", "10.1.2 ", "(a) "
    numbered_pattern = r'\n\s*(?:\d+\.[\d.]*\s|\([a-zA-Z0-9]+\)\s)'
    parts = re.split(numbered_pattern, text)

    if len(parts) <= 1:
        # Strategy 2 (fallback): Split on double-newline paragraph breaks
        parts = re.split(r'\n\s*\n', text)

    # Filter out short fragments (likely headers, section numbers, noise)
    clauses = [clause.strip() for clause in parts if len(clause.strip()) >= 20]

    logger.info("Split text into %d clauses (from %d raw parts)",
                len(clauses), len(parts))
    return clauses
