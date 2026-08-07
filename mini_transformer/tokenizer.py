"""Word-level tokenizer and vocabulary for the mini-Transformer.

This module provides:
- load_clause_records: Load cached clause records from Phase 1's JSON
- basic_tokenize: Regex-based word-level tokenizer
- Vocabulary: Build, encode, decode, save, load a word-level vocabulary
- build_and_save_vocab: End-to-end vocab building from clause records
"""

import json
import logging
import re
from collections import Counter
from typing import Dict, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Data loading (reuses Phase 1 cached JSON — no CUAD re-download)
# ---------------------------------------------------------------------------


def load_clause_records(path: str = "data/processed/clause_records.json") -> List[Dict]:
    """Load cached clause records produced by Phase 1.

    Args:
        path: Path to the JSON file containing clause records.

    Returns:
        List of dicts, each with keys like ``clause_text``,
        ``risk_label``, ``clause_type``, etc.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Clause records not found at '{path}'. "
            f"Run Phase 1 first (python -m src.dataset) to generate this file."
        )
    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    logger.info("Loaded %d clause records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Basic word-level tokenizer
# ---------------------------------------------------------------------------


def basic_tokenize(text: str) -> List[str]:
    r"""Tokenize text into lowercase word-level tokens.

    Uses ``re.findall(r"\b\w+\b", ...)`` to extract alphanumeric
    sequences. Punctuation is stripped (not kept as separate tokens).

    Args:
        text: Input text string.

    Returns:
        List of lowercase token strings.
    """
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Vocabulary:
    """Word-level vocabulary with fixed special token ids.

    Special tokens and their FIXED ids (never reassigned during build):
        PAD = 0, UNK = 1, CLS = 2, SEP = 3
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"
    SEP_TOKEN = "<SEP>"

    # Fixed ids — these must never shift
    PAD_ID = 0
    UNK_ID = 1
    CLS_ID = 2
    SEP_ID = 3

    def __init__(self, max_vocab_size: int = 15000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Pre-register special tokens at their fixed ids
        self.token2id: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.CLS_TOKEN: self.CLS_ID,
            self.SEP_TOKEN: self.SEP_ID,
        }
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}

        # Stored after build() for reporting
        self._token_counts: Counter = Counter()

    def build(self, texts: List[str]) -> None:
        """Build vocabulary from a corpus of texts.

        Tokenizes every text, counts frequencies across the entire
        corpus, keeps tokens with freq >= min_freq, and takes the
        top (max_vocab_size - 4) most frequent tokens (leaving room
        for the 4 special tokens).

        Args:
            texts: List of raw text strings (e.g. clause_text values).
        """
        self._token_counts = Counter()
        for text in texts:
            tokens = basic_tokenize(text)
            self._token_counts.update(tokens)

        logger.info("Total unique tokens before filtering: %d", len(self._token_counts))

        # Filter by min_freq
        filtered = {tok: cnt for tok, cnt in self._token_counts.items()
                    if cnt >= self.min_freq}
        logger.info("Tokens with freq >= %d: %d", self.min_freq, len(filtered))

        # Sort by frequency descending, take top (max_vocab_size - 4)
        num_special = 4
        max_regular = self.max_vocab_size - num_special
        sorted_tokens = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:max_regular]

        # Assign sequential ids starting from 4
        for idx, (token, _count) in enumerate(top_tokens):
            token_id = num_special + idx
            self.token2id[token] = token_id
            self.id2token[token_id] = token

        logger.info("Final vocabulary size: %d (including %d special tokens)",
                     len(self.token2id), num_special)

        # Log top 10 most frequent tokens with real counts
        top_10 = sorted_tokens[:10]
        logger.info("Top 10 tokens by frequency:")
        for token, count in top_10:
            logger.info("  %-20s  freq=%d", token, count)

    def encode(
        self, text: str, max_length: int, add_special_tokens: bool = True
    ) -> List[int]:
        """Encode text into a fixed-length list of token ids.

        Args:
            text: Input text string.
            max_length: Exact output length (with padding).
            add_special_tokens: If True, prepend CLS and append SEP.

        Returns:
            List of int ids, length == max_length exactly.
        """
        tokens = basic_tokenize(text)

        # Truncate to leave room for special tokens if needed
        if add_special_tokens:
            max_content = max_length - 2  # room for CLS + SEP
        else:
            max_content = max_length
        tokens = tokens[:max_content]

        # Map to ids (UNK for unknown tokens)
        ids = [self.token2id.get(tok, self.UNK_ID) for tok in tokens]

        # Add special tokens
        if add_special_tokens:
            ids = [self.CLS_ID] + ids + [self.SEP_ID]

        # Pad to exactly max_length
        padding_needed = max_length - len(ids)
        ids = ids + [self.PAD_ID] * padding_needed

        assert len(ids) == max_length, (
            f"encode() produced {len(ids)} ids, expected {max_length}"
        )
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back to a text string.

        PAD tokens are skipped in the output. This is for debugging
        and sanity checks.

        Args:
            ids: List of integer token ids.

        Returns:
            Space-joined string of decoded tokens (excluding PAD).
        """
        tokens = []
        for token_id in ids:
            if token_id == self.PAD_ID:
                continue  # skip padding
            token = self.id2token.get(token_id, self.UNK_TOKEN)
            tokens.append(token)
        return " ".join(tokens)

    def __len__(self) -> int:
        """Return actual vocabulary size (including special tokens)."""
        return len(self.token2id)

    def save(self, path: str) -> None:
        """Save token2id mapping as JSON to disk.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.token2id, fh, indent=2, ensure_ascii=False)
        logger.info("Vocabulary saved to %s (%d tokens)", path, len(self.token2id))

    def load(self, path: str) -> None:
        """Load token2id mapping from a JSON file and rebuild id2token.

        Args:
            path: Path to the saved vocabulary JSON.
        """
        with open(path, "r", encoding="utf-8") as fh:
            self.token2id = json.load(fh)
        # JSON keys are strings; values (ids) are ints — rebuild reverse map
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        logger.info("Vocabulary loaded from %s (%d tokens)", path, len(self.token2id))


# ---------------------------------------------------------------------------
# End-to-end vocab builder
# ---------------------------------------------------------------------------


def build_and_save_vocab(
    records_path: str = "data/processed/clause_records.json",
    output_path: str = "mini_transformer/vocab.json",
    max_vocab_size: int = 15000,
    min_freq: int = 2,
) -> Vocabulary:
    """Build a vocabulary from cached clause records and save to disk.

    Args:
        records_path: Path to the Phase 1 clause records JSON.
        output_path: Path to save the vocabulary JSON.
        max_vocab_size: Maximum vocabulary size (including specials).
        min_freq: Minimum token frequency to include.

    Returns:
        The built Vocabulary instance.
    """
    records = load_clause_records(records_path)
    texts = [r["clause_text"] for r in records]
    logger.info("Building vocabulary from %d clause texts …", len(texts))

    vocab = Vocabulary(max_vocab_size=max_vocab_size, min_freq=min_freq)
    vocab.build(texts)
    vocab.save(output_path)

    return vocab


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build vocab from the real cached dataset
    vocab = build_and_save_vocab()

    print(f"\nVocabulary size: {len(vocab)}")

    # Top 10 most frequent tokens with real counts
    top_10 = vocab._token_counts.most_common(10)
    print("\nTop 10 tokens by frequency:")
    for token, count in top_10:
        print(f"  {token:<20s}  freq={count}")

    # Load one real clause for encode/decode roundtrip
    records = load_clause_records()
    sample_text = records[0]["clause_text"]
    print(f"\nSample clause (first 120 chars): {sample_text[:120]}...")

    # Encode
    encoded = vocab.encode(sample_text, max_length=64, add_special_tokens=True)
    print(f"\nEncoded (max_length=64): {encoded}")
    print(f"Encoded length: {len(encoded)} (expected: 64)")
    assert len(encoded) == 64, f"Length mismatch: {len(encoded)} != 64"

    # Decode
    decoded = vocab.decode(encoded)
    print(f"\nDecoded: {decoded}")

    # Verify vocab.json was saved
    import os
    vocab_path = "mini_transformer/vocab.json"
    if os.path.exists(vocab_path):
        size_kb = os.path.getsize(vocab_path) / 1024
        print(f"\nvocab.json saved: {os.path.abspath(vocab_path)} ({size_kb:.1f} KB)")
    else:
        print(f"\n❌ vocab.json NOT found at {vocab_path}")

    print("\n[OK] Phase A tokenizer working.")
