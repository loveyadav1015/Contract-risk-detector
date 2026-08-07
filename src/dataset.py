"""Dataset loading and preprocessing for the Contract Risk Detector.

This module handles:
- Loading the CUAD dataset from HuggingFace Hub
- Mapping CUAD SQuAD-format questions to clause-type categories
- Converting raw examples into structured clause records with risk labels
- Stratified train / val / test splitting
- Building a PyTorch Dataset that tokenizes clauses with sliding-window chunking
- Constructing DataLoaders ready for training and evaluation
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.utils import (
    chunk_text,
    clean_text,
    flatten_cuad_answers,
    get_logger,
    get_risk_label_id,
    get_risk_level,
    load_config,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CUAD question-title → clause-type mapping  (all 41 categories)
# ---------------------------------------------------------------------------
# Keys are the category names that appear inside the CUAD question strings
# (between quotation marks).  Values are the clause-type strings used
# throughout this project, or ``None`` for metadata questions that should be
# skipped.
#
# Note: CUAD uses "Uncapped Liability" — we map it to "Unlimited Liability"
# to match the risk-level definitions in ``src/utils.py``.
# ---------------------------------------------------------------------------

CUAD_QUESTION_TO_CLAUSE: Dict[str, Optional[str]] = {
    # ── Metadata (skip) ────────────────────────────────────────────────
    "Document Name": None,
    "Parties": None,
    "Agreement Date": None,
    "Effective Date": None,
    "Expiration Date": None,
    # ── Clause types ───────────────────────────────────────────────────
    "Renewal Term": "Renewal Term",
    "Notice Period To Terminate Renewal": "Notice Period To Terminate Renewal",
    "Governing Law": "Governing Law",
    "Most Favored Nation": "Most Favored Nation",
    "Non-Compete": "Non-Compete",
    "Exclusivity": "Exclusivity",
    "No-Solicit Of Customers": "No-Solicit Of Customers",
    "Competitive Restriction Exception": "Competitive Restriction Exception",
    "No-Solicit Of Employees": "No-Solicit Of Employees",
    "Non-Disparagement": "Non-Disparagement",
    "Termination For Convenience": "Termination For Convenience",
    "Rofr/Rofo/Rofn": "Rofr/Rofo/Rofn",
    "Change Of Control": "Change Of Control",
    "Anti-Assignment": "Anti-Assignment",
    "Revenue/Profit Sharing": "Revenue/Profit Sharing",
    "Price Restrictions": "Price Restrictions",
    "Minimum Commitment": "Minimum Commitment",
    "Volume Restriction": "Volume Restriction",
    "IP Ownership Assignment": "IP Ownership Assignment",
    "Joint IP Ownership": "Joint IP Ownership",
    "License Grant": "License Grant",
    "Non-Transferable License": "Non-Transferable License",
    "Affiliate License-Licensor": "Affiliate License-Licensor",
    "Affiliate License-Licensee": "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License": "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License": "Irrevocable Or Perpetual License",
    "Source Code Escrow": "Source Code Escrow",
    "Post-Termination Services": "Post-Termination Services",
    "Audit Rights": "Audit Rights",
    "Uncapped Liability": "Unlimited Liability",
    "Cap On Liability": "Cap On Liability",
    "Liquidated Damages": "Liquidated Damages",
    "Warranty Duration": "Warranty Duration",
    "Insurance": "Insurance",
    "Covenant Not To Sue": "Covenant Not To Sue",
    "Third Party Beneficiary": "Third Party Beneficiary",
}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _extract_question_title(question: str) -> str:
    """Extract the category name from a CUAD question string.

    CUAD questions follow the pattern::

        Highlight the parts … related to "Category Name" that should be …

    Returns the text between the first pair of double-quote characters,
    or an empty string if no match is found.
    """
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1)
    # Fallback: try Unicode left/right double quotation marks
    match = re.search(r"\u201c([^\u201d]+)\u201d", question)
    if match:
        return match.group(1)
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


# NOTE: datasets library is pinned to ==2.19.0 in requirements.txt because
# v4+/v5+ dropped support for script-based dataset loading, which the CUAD
# repository on HuggingFace Hub still uses.
def load_cuad_raw(cache_dir: str = "data/raw") -> DatasetDict:
    """Load the CUAD dataset from HuggingFace Hub.

    Args:
        cache_dir: Local directory to download and cache the dataset in.

    Returns:
        A ``DatasetDict`` with at least one split (typically ``"test"``).
    """
    logger.info("Downloading CUAD dataset to: %s", cache_dir)
    dataset = load_dataset("theatticusproject/cuad-qa", cache_dir=cache_dir)
    logger.info("CUAD dataset loaded. Splits: %s", list(dataset.keys()))
    # CUAD has a single "test" split with ~22 450 examples
    for split_name, split_data in dataset.items():
        logger.info("  %s: %d examples", split_name, len(split_data))
    return dataset


def cuad_to_clause_records(dataset: DatasetDict) -> List[Dict]:
    """Convert raw CUAD examples into structured clause records.

    Iterates over the available split (``"train"`` if present, otherwise
    ``"test"``), extracts clause text via :func:`flatten_cuad_answers`,
    cleans it, and assigns risk labels.

    Each returned record is a dict with keys:
    ``contract_id``, ``clause_type``, ``clause_text``,
    ``risk_level``, ``risk_label``.
    """
    # CUAD ships a single "test" split; fall back gracefully
    if "train" in dataset:
        split = dataset["train"]
    elif "test" in dataset:
        split = dataset["test"]
    else:
        raise ValueError(
            f"No usable split found in dataset. Available: {list(dataset.keys())}"
        )
    logger.info("Processing split with %d raw examples …", len(split))

    records: List[Dict] = []
    skipped_no_clause = 0
    skipped_no_answer = 0
    skipped_too_short = 0

    for example in split:
        # --- Map question → clause type ---
        title = _extract_question_title(example["question"])
        if title not in CUAD_QUESTION_TO_CLAUSE:
            skipped_no_clause += 1
            continue
        clause_type = CUAD_QUESTION_TO_CLAUSE[title]
        if clause_type is None:
            skipped_no_clause += 1
            continue

        # --- Extract answer text ---
        clause_text, is_present = flatten_cuad_answers(example["answers"])
        if not is_present:
            skipped_no_answer += 1
            continue

        # --- Clean ---
        clause_text = clean_text(clause_text)
        if len(clause_text) < 10:
            skipped_too_short += 1
            continue

        records.append(
            {
                "contract_id": example.get("title", example.get("id", "")),
                "clause_type": clause_type,
                "clause_text": clause_text,
                "risk_level": get_risk_level(clause_type),
                "risk_label": get_risk_label_id(clause_type),
            }
        )

    # --- Log summary ---
    logger.info("Total clause records: %d", len(records))
    logger.info(
        "Skipped — no clause type: %d | no answer: %d | too short: %d",
        skipped_no_clause,
        skipped_no_answer,
        skipped_too_short,
    )
    dist = Counter(r["risk_level"] for r in records)
    logger.info(
        "Class distribution — low: %d | medium: %d | high: %d",
        dist.get("low", 0),
        dist.get("medium", 0),
        dist.get("high", 0),
    )
    return records


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_records(
    records: List[Dict],
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split of clause records into train / val / test.

    The test set is split off first, then the validation set is carved
    from the remainder so that each final set respects the requested
    proportions relative to the full dataset.
    """
    labels = [r["risk_label"] for r in records]

    # 1. Split off the test set
    train_val, test = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    # 2. Split val from the remainder
    train_val_labels = [r["risk_label"] for r in train_val]
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class ContractClauseDataset(Dataset):
    """PyTorch Dataset that tokenizes clause records with sliding-window chunking.

    Each record is chunked via :func:`~src.utils.chunk_text` and then
    tokenized.  A long clause therefore produces multiple samples, each
    sharing the same label.
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 512,
        stride: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.samples: List[Dict[str, torch.Tensor]] = self._build_samples(records)

    def _build_samples(self, records: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        """Chunk, tokenize, and assemble every record into model-ready dicts."""
        samples: List[Dict[str, torch.Tensor]] = []
        # Reserve 2 tokens for [CLS] and [SEP]
        effective_max_tokens = self.max_length - 2

        for record in records:
            chunks = chunk_text(
                record["clause_text"],
                self.tokenizer,
                max_tokens=effective_max_tokens,
                stride=self.stride,
            )
            for chunk in chunks:
                encoded = self.tokenizer(
                    chunk,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                sample: Dict[str, torch.Tensor] = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": torch.tensor(record["risk_label"], dtype=torch.long),
                }
                # LegalBERT returns token_type_ids; DeBERTa does not.
                if "token_type_ids" in encoded:
                    sample["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_dataloaders(
    train_records: List[Dict],
    val_records: List[Dict],
    test_records: List[Dict],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for train / val / test splits."""
    train_ds = ContractClauseDataset(train_records, tokenizer, max_length, stride)
    val_ds = ContractClauseDataset(val_records, tokenizer, max_length, stride)
    test_ds = ContractClauseDataset(test_records, tokenizer, max_length, stride)

    logger.info(
        "Dataset sizes (after chunking) — train: %d | val: %d | test: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# End-to-end pipelines
# ---------------------------------------------------------------------------


def _load_records(config: Dict) -> List[Dict]:
    """Load clause records, using the processed cache if available."""
    raw_data_dir: str = config["paths"]["raw_data_dir"]
    processed_data_dir: str = config["paths"]["processed_data_dir"]
    os.makedirs(processed_data_dir, exist_ok=True)
    cached_records_path = os.path.join(processed_data_dir, "clause_records.json")

    if os.path.exists(cached_records_path):
        logger.info("Loading cached clause records from %s", cached_records_path)
        with open(cached_records_path, "r", encoding="utf-8") as fh:
            records = json.load(fh)
        logger.info("Loaded %d cached records.", len(records))
    else:
        dataset = load_cuad_raw(cache_dir=raw_data_dir)
        records = cuad_to_clause_records(dataset)
        with open(cached_records_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False)
        logger.info("Saved %d clause records to %s", len(records), cached_records_path)

    return records


def build_datasets(
    config: Optional[Dict] = None,
) -> Tuple["ContractClauseDataset", "ContractClauseDataset", "ContractClauseDataset", AutoTokenizer]:
    """Build Dataset objects for HuggingFace Trainer.

    Unlike :func:`build_data_pipeline` this returns the raw
    ``ContractClauseDataset`` splits (not DataLoaders), which is what
    HuggingFace Trainer expects.

    Returns:
        ``(train_ds, val_ds, test_ds, tokenizer)``
    """
    if config is None:
        config = load_config()

    model_name: str = config["model"]["name"]
    max_length: int = config["model"]["max_length"]
    val_size: float = config["data"]["val_size"]
    test_size: float = config["data"]["test_size"]
    stride: int = config["data"]["stride"]
    seed: int = config["training"]["seed"]

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    records = _load_records(config)
    train_rec, val_rec, test_rec = split_records(
        records, val_size=val_size, test_size=test_size, seed=seed
    )

    train_ds = ContractClauseDataset(train_rec, tokenizer, max_length, stride)
    val_ds = ContractClauseDataset(val_rec, tokenizer, max_length, stride)
    test_ds = ContractClauseDataset(test_rec, tokenizer, max_length, stride)

    logger.info(
        "Dataset sizes (after chunking) — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds, tokenizer


def build_data_pipeline(
    config: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """Build the complete data pipeline from config to DataLoaders.

    Wraps :func:`build_datasets` with DataLoaders for use outside of
    HuggingFace Trainer (e.g. custom training loops, smoke tests).

    Returns:
        ``(train_loader, val_loader, test_loader, tokenizer)``
    """
    if config is None:
        config = load_config()

    batch_size: int = config["training"]["batch_size"]
    num_workers: int = config["training"]["num_workers"]

    train_ds, val_ds, test_ds, tokenizer = build_datasets(config)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, tokenizer


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = build_data_pipeline()

    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"Labels: {batch['labels']}")
    print("✅ Phase 1 pipeline working.")