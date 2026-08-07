"""Transformer encoder: feed-forward, encoder block, and stacked encoder."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mini_transformer.attention import MultiHeadSelfAttention, create_padding_mask
from mini_transformer.embeddings import TransformerEmbedding
from mini_transformer.tokenizer import Vocabulary, load_clause_records


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Dropout → Linear."""

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # GELU: smoother gradients than ReLU; common in modern Transformers.
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderBlock(nn.Module):
    """One encoder layer: self-attention and FFN, each with residual + LayerNorm.

    Post-LN (Vaswani et al.): sublayer output is added to the residual stream,
    then LayerNorm is applied — not Pre-LN (norm before sublayer).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        attn_output, attn_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Token embedding + stack of encoder blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_len, dropout
        )
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, token_ids: Tensor, mask: Tensor | None = None
    ) -> Tuple[Tensor, List[Tensor]]:
        x = self.embedding(token_ids)
        attn_weights_all_layers: List[Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_all_layers.append(attn_weights)
        return x, attn_weights_all_layers


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    d_model = 256
    max_len = 64

    vocab = Vocabulary()
    vocab.load("mini_transformer/vocab.json")

    records = load_clause_records()
    sample_texts = [records[i]["clause_text"] for i in range(4)]
    encoded = [vocab.encode(text, max_length=max_len) for text in sample_texts]
    token_ids = torch.tensor(encoded, dtype=torch.long)
    mask = create_padding_mask(token_ids, pad_token_id=vocab.PAD_ID)

    encoder = TransformerEncoder(
        vocab_size=len(vocab),
        d_model=256,
        num_heads=4,
        d_ff=1024,
        num_layers=4,
        max_len=max_len,
    )
    encoder.eval()

    with torch.no_grad():
        output, attn_weights_all_layers = encoder(token_ids, mask)

    print(f"token_ids.shape: {tuple(token_ids.shape)}")
    print(f"output.shape: {tuple(output.shape)}")
    assert tuple(output.shape) == (4, max_len, d_model)

    print(f"len(attn_weights_all_layers): {len(attn_weights_all_layers)}")
    assert len(attn_weights_all_layers) == 4
    for i, aw in enumerate(attn_weights_all_layers):
        assert tuple(aw.shape) == (4, 4, max_len, max_len), f"layer {i} attn shape"

    has_nan = torch.isnan(output).any().item()
    print(f"torch.isnan(output).any(): {has_nan}")

    trainable_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    )
    print(f"total trainable parameters: {trainable_params}")

    if not has_nan:
        print("✅ Phase 4 (mini-transformer) encoder block working.")
