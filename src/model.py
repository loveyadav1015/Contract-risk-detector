"""Model architecture and classification head for the Contract Risk Detector.

This module provides:
- ContractRiskClassifier: Custom classifier built on a pretrained
  transformer (LegalBERT / DeBERTa) with a linear head over [CLS]
- build_model: Factory function to instantiate and place on device
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.utils import get_logger

logger = get_logger(__name__)


class ContractRiskClassifier(nn.Module):
    """Transformer + linear classification head for clause risk prediction.

    Architecture:
        base_model  →  [CLS] hidden state  →  Dropout  →  Linear(num_labels)

    Forward returns a ``SequenceClassifierOutput`` so it works seamlessly
    with HuggingFace Trainer (loss is computed internally when *labels*
    are provided).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> SequenceClassifierOutput:
        """Run the forward pass and optionally compute CrossEntropyLoss.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Segment IDs (passed only if not None — LegalBERT
                uses them, DeBERTa does not).
            labels: Ground-truth class indices.  When provided the loss is
                computed with ``CrossEntropyLoss`` and returned inside the
                output object.

        Returns:
            ``SequenceClassifierOutput`` with ``.loss`` (if labels given)
            and ``.logits`` (raw, no softmax — CrossEntropyLoss expects this).
        """
        # Only pass token_type_ids when the base model supports it
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.base_model(**kwargs)

        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(config: dict) -> ContractRiskClassifier:
    """Instantiate a ContractRiskClassifier from *config* and place it on
    the best available device (CUDA if present, else CPU).
    """
    model_name: str = config["model"]["name"]
    num_labels: int = config["model"]["num_labels"]
    dropout: float = config["model"]["dropout"]

    model = ContractRiskClassifier(model_name, num_labels, dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model built — %s | %d labels | device: %s",
                model_name, num_labels, device)

    return model
