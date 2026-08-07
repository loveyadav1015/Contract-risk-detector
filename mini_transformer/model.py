"""Mini-transformer classifier head on top of the encoder [CLS] output."""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from mini_transformer.encoder import TransformerEncoder


class MiniTransformerClassifier(nn.Module):
    """Transformer encoder + CLS classification head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
        max_len: int = 64,
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, token_ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        encoder_output, _ = self.encoder(token_ids, mask)   # (batch, seq_len, d_model)
        cls_representation = encoder_output[:, 0, :]         # (batch, d_model)
        logits = self.classifier(self.dropout(cls_representation))  # (batch, num_labels)
        return logits