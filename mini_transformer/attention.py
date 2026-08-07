"""Multi-head scaled dot-product self-attention (from scratch)."""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mini_transformer.embeddings import TransformerEmbedding
from mini_transformer.tokenizer import Vocabulary, load_clause_records


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with separate Q/K/V/O linear projections per head."""

    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: Tensor, batch_size: int) -> Tensor:
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def scaled_dot_product_attention(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        # mask: (batch_size, 1, 1, seq_len) — 1 = attend, 0 = pad key (masked)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        Q = self.split_heads(self.W_q(x), batch_size)
        K = self.split_heads(self.W_k(x), batch_size)
        V = self.split_heads(self.W_v(x), batch_size)
        attn_output, attn_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        combined = self.combine_heads(attn_output, batch_size)
        output = self.W_o(combined)
        return output, attn_weights


def create_padding_mask(token_ids: Tensor, pad_token_id: int = 0) -> Tensor:
    """Padding mask for attention scores.

    Returns float tensor of shape (batch_size, 1, 1, seq_len):
    1.0 where token is not PAD, 0.0 at PAD positions (for masked_fill).
    """
    mask = (token_ids != pad_token_id).float()
    return mask.unsqueeze(1).unsqueeze(2)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    d_model = 256
    num_heads = 4
    max_len = 64

    vocab = Vocabulary()
    vocab.load("mini_transformer/vocab.json")

    records = load_clause_records()
    sample_texts = [records[i]["clause_text"] for i in range(4)]
    encoded = [vocab.encode(text, max_length=max_len) for text in sample_texts]
    token_ids = torch.tensor(encoded, dtype=torch.long)

    embed = TransformerEmbedding(
        vocab_size=len(vocab), d_model=d_model, max_len=max_len
    )
    attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    embed.eval()
    attention.eval()

    with torch.no_grad():
        embeddings = embed(token_ids)
        pad_mask = create_padding_mask(token_ids, pad_token_id=vocab.PAD_ID)
        output, attn_weights = attention(embeddings, mask=pad_mask)

    print(f"embeddings.shape: {tuple(embeddings.shape)}")
    print(f"output.shape: {tuple(output.shape)}")
    print(f"attn_weights.shape: {tuple(attn_weights.shape)}")
    assert tuple(output.shape) == tuple(embeddings.shape), (
        "Output shape must match input for residual connections"
    )

    has_nan = torch.isnan(output).any().item()
    print(f"torch.isnan(output).any(): {has_nan}")

    row_sum = attn_weights[0, 0, 0, :].sum().item()
    print(f"attn_weights[0, 0, 0, :].sum(): {row_sum}")
    sum_ok = abs(row_sum - 1.0) < 1e-4
    print(f"sum close to 1.0 (within 1e-4): {sum_ok}")

    pad_positions = (token_ids[0] == vocab.PAD_ID).nonzero(as_tuple=True)[0]
    first_pad = pad_positions[0].item() if len(pad_positions) > 0 else max_len - 1
    pad_attn_query0 = attn_weights[0, 0, 0, first_pad].item()
    pad_attn_query10 = attn_weights[0, 0, 10, first_pad].item()
    print(f"first PAD key index in sample 0: {first_pad}")
    print(
        f"attn_weights[0, 0, 0, pad_key={first_pad}]: {pad_attn_query0}"
    )
    print(
        f"attn_weights[0, 0, 10, pad_key={first_pad}]: {pad_attn_query10}"
    )
    pad_near_zero = pad_attn_query0 < 1e-6 and pad_attn_query10 < 1e-6
    print(f"PAD key positions have near-zero weight: {pad_near_zero}")

    if not has_nan and sum_ok and pad_near_zero:
        print("✅ Phase 3 (mini-transformer) multi-head self-attention working.")
