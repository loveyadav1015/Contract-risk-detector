"""Token and sinusoidal positional embeddings for the mini-transformer."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from mini_transformer.tokenizer import Vocabulary, load_clause_records


class TokenEmbedding(nn.Module):
    """Lookup token ids into d_model vectors with sqrt(d_model) scaling."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Token embeddings followed by positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embedding(token_ids)
        return self.positional_encoding(x)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    d_model = 256
    max_len = 64

    vocab = Vocabulary()
    vocab.load("mini_transformer/vocab.json")
    assert vocab.token2id[vocab.PAD_TOKEN] == 0, "PAD token id must be 0 for padding_idx"

    records = load_clause_records()
    sample_texts = [records[i]["clause_text"] for i in range(4)]
    encoded = [vocab.encode(text, max_length=max_len) for text in sample_texts]
    token_ids = torch.tensor(encoded, dtype=torch.long)

    model = TransformerEmbedding(
        vocab_size=len(vocab), d_model=d_model, max_len=max_len
    )
    model.eval()

    with torch.no_grad():
        output = model(token_ids)

    print(f"token_ids.shape: {tuple(token_ids.shape)}")
    print(f"output.shape: {tuple(output.shape)}")
    print(f"output[0, 0, :5]: {output[0, 0, :5].tolist()}")

    has_nan = torch.isnan(output).any().item()
    print(f"torch.isnan(output).any(): {has_nan}")

    token_emb = TokenEmbedding(vocab_size=len(vocab), d_model=d_model)
    token_emb.eval()
    pad_ids = torch.zeros(2, max_len, dtype=torch.long)
    with torch.no_grad():
        pad_token_emb = token_emb(pad_ids)
    pad_all_zero = torch.all(pad_token_emb == 0).item()
    print(f"TokenEmbedding on all-PAD ids all zero: {pad_all_zero}")

    if (
        tuple(output.shape) == (4, 64, 256)
        and not has_nan
        and pad_all_zero
    ):
        print("✅ Phase 2 (mini-transformer) embeddings working.")
