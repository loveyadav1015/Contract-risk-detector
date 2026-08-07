"""Mini-transformer prediction service for the Contract Risk Detector API."""

import os
from typing import Dict

import torch

from api.services.predictor import LABEL_NAMES, RiskPredictor
from mini_transformer.attention import create_padding_mask
from mini_transformer.model import MiniTransformerClassifier
from mini_transformer.tokenizer import Vocabulary
from src.utils import clean_text, get_logger

logger = get_logger(__name__)


class MiniTransformerPredictor:
    """Loads a trained mini-transformer checkpoint and runs inference."""

    def __init__(
        self,
        model_path: str = "mini_transformer/best_model_v2",
        device: str = None,
    ):
        self.max_len = 64

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Loading mini-transformer vocab from %s", model_path)
        self.vocab = Vocabulary()
        self.vocab.load(os.path.join(model_path, "vocab.json"))

        logger.info("Reconstructing MiniTransformerClassifier for inference.")
        self.model = MiniTransformerClassifier(
            vocab_size=len(self.vocab),
            d_model=256,
            num_heads=4,
            d_ff=1024,
            num_layers=4,
            max_len=self.max_len,
            num_labels=3,
            dropout=0.1,
        )

        state_dict = torch.load(
            os.path.join(model_path, "mini_transformer_best.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("MiniTransformerPredictor ready on device: %s", self.device)

    def predict(self, text: str) -> Dict:
        """Run risk prediction on one clause."""
        normalized = clean_text(text)
        if not normalized:
            return {
                "risk_level": "low",
                "risk_label": 0,
                "confidence": 0.0,
                "model": "mini_transformer_from_scratch",
            }

        token_ids = self.vocab.encode(
            normalized,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mask = create_padding_mask(input_ids, pad_token_id=self.vocab.PAD_ID).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, mask=mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            risk_label = int(torch.argmax(probs).item())
            confidence = float(probs[risk_label].item())

        return {
            "risk_level": LABEL_NAMES[risk_label],
            "risk_label": risk_label,
            "confidence": round(confidence, 4),
            "model": "mini_transformer_from_scratch",
        }

    @staticmethod
    def get_risk_reason(risk_label: int) -> str:
        """Reuse LegalBERT service heuristic risk explanation text."""
        return RiskPredictor.get_risk_reason(risk_label)
