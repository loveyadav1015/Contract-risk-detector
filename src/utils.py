"""Helper functions for the Contract Risk Detector.

This module provides utilities for:
- Risk level mapping from CUAD clause types
- Text cleaning and normalization
- Clause chunking (splitting long texts into model-sized segments via tokenizer)
- CUAD answer flattening
- Configuration loading from YAML
- Logging setup
"""

import logging
import os
import re
from typing import Dict, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# Risk-level mappings
# ---------------------------------------------------------------------------

HIGH_RISK_CLAUSES = {
    "Indemnification",
    "Non-Compete",
    "Exclusivity",
    "IP Ownership Assignment",
    "Liquidated Damages",
    "Unlimited Liability",
    "Change Of Control",
    "Anti-Assignment",
    "Most Favored Nation",
    "Non-Disparagement",
}

MEDIUM_RISK_CLAUSES = {
    "Termination For Convenience",
    "Governing Law",
    "Dispute Resolution",
    "Audit Rights",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Cap On Liability",
    "Warranty Duration",
    "Insurance",
}


def get_risk_level(clause_type: str) -> str:
    """Map a CUAD clause type to a risk level string.

    Args:
        clause_type: The CUAD clause type name.

    Returns:
        "high", "medium", or "low".
    """
    if clause_type in HIGH_RISK_CLAUSES:
        return "high"
    if clause_type in MEDIUM_RISK_CLAUSES:
        return "medium"
    return "low"


def get_risk_label_id(clause_type: str) -> int:
    """Map a CUAD clause type to a numeric risk label.

    Returns:
        2 for high, 1 for medium, 0 for low.
    """
    level = get_risk_level(clause_type)
    if level == "high":
        return 2
    if level == "medium":
        return 1
    return 0


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Clean and normalize text.

    - Removes non-printable / control characters (U+0000–U+0008, U+000B,
      U+000C, U+000E–U+001F, U+007F–U+009F)
    - Collapses multiple whitespace characters into a single space
    - Strips leading / trailing whitespace
    - Returns "" for non-string input
    """
    if not isinstance(text, str):
        return ""
    # Remove non-printable / control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse all whitespace (newlines, tabs, spaces) into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 512,
    stride: int = 128,
) -> List[str]:
    """Split *text* into overlapping chunks based on token count.

    Uses *tokenizer* to tokenize without special tokens.  If the full
    token count fits within *max_tokens*, the original text is returned
    as a single-element list.  Otherwise a sliding window with *stride*
    overlap is applied and each window is decoded back to a string.

    Args:
        text: The input text to chunk.
        tokenizer: A HuggingFace tokenizer instance.
        max_tokens: Maximum number of tokens per chunk.
        stride: Number of tokens that consecutive chunks overlap by.

    Returns:
        A list of text-chunk strings.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return [text]

    step = max_tokens - stride
    if step <= 0:
        # Guard against bad config: fall back to non-overlapping chunks
        step = max_tokens

    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_str)
        if end >= len(token_ids):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# CUAD helpers
# ---------------------------------------------------------------------------


def flatten_cuad_answers(answers: Dict) -> Tuple[str, bool]:
    """Extract the first answer from the CUAD answers format.

    CUAD answers follow the SQuAD convention:
    ``{"text": [...], "answer_start": [...]}``.

    Returns:
        ``(first_text.strip(), True)`` if a non-empty answer exists,
        otherwise ``("", False)``.
    """
    texts = answers.get("text", [])
    if texts and isinstance(texts[0], str) and texts[0].strip():
        return (texts[0].strip(), True)
    return ("", False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load a YAML configuration file and return it as a dict.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a logger with a formatted StreamHandler.

    Format: ``YYYY-MM-DD HH:MM:SS | LEVEL | name | message``

    Duplicate handlers are avoided when calling this function more than
    once with the same *name*.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
