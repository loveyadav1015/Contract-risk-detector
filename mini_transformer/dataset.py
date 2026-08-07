"""Dataset and split helpers for mini-transformer training."""

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from mini_transformer.tokenizer import Vocabulary, load_clause_records
from src.dataset import split_records


class MiniTransformerDataset(Dataset):
    """PyTorch dataset for fixed-length token ids and risk labels."""

    def __init__(
        self, records: List[Dict], vocab: Vocabulary, max_length: int = 64
    ):
        self.records = records
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        token_ids = self.vocab.encode(
            record["clause_text"],
            self.max_length,
            add_special_tokens=True,
        )
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label": torch.tensor(record["risk_label"], dtype=torch.long),
        }


def load_splits_from_existing_pipeline(
    config: Dict,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Reuse src.dataset split logic and config values for exact split parity."""
    processed_data_dir = config["paths"]["processed_data_dir"]
    records_path = os.path.join(processed_data_dir, "clause_records.json")
    all_records = load_clause_records(records_path)

    train_records, val_records, test_records = split_records(
        all_records,
        val_size=config["data"]["val_size"],
        test_size=config["data"]["test_size"],
        seed=config["training"]["seed"],
    )
    return train_records, val_records, test_records
