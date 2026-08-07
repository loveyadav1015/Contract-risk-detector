"""Risk prediction service for the Contract Risk Detector API."""

import os
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
from typing import Dict

from src.model import ContractRiskClassifier
from src.utils import (
    HIGH_RISK_CLAUSES,
    MEDIUM_RISK_CLAUSES,
    chunk_text,
    clean_text,
    get_logger,
    load_config,
)

logger = get_logger(__name__)

LABEL_NAMES = {0: "low", 1: "medium", 2: "high"}


class RiskPredictor:
    """Loads a trained ContractRiskClassifier and runs inference."""

    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/config.yaml",
        device: str = None,
    ):
        self.config = load_config(config_path)

        # Read architecture params from config (same values used during training)
        model_name: str = self.config["model"]["name"]
        num_labels: int = self.config["model"]["num_labels"]
        dropout: float = self.config["model"]["dropout"]
        self.max_length: int = self.config["model"]["max_length"]
        self.stride: int = self.config["data"]["stride"]

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model_source = model_path
        if os.path.exists(model_path):
            logger.info("Using local LegalBERT checkpoint at %s", model_path)
        else:
            hf_repo_id = os.getenv("HF_LEGALBERT_REPO_ID")
            if not hf_repo_id:
                raise FileNotFoundError(
                    f"Local model path does not exist: {model_path}. "
                    "Set HF_LEGALBERT_REPO_ID to a HuggingFace model repo "
                    "(for example: your-username/contract-risk-legalbert) "
                    "to enable remote checkpoint loading."
                )

            logger.warning(
                "Local model path %s not found. Falling back to HuggingFace Hub repo: %s",
                model_path,
                hf_repo_id,
            )
            model_source = snapshot_download(
                repo_id=hf_repo_id,
                allow_patterns=[
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                ],
            )
            logger.info("Downloaded LegalBERT checkpoint snapshot to %s", model_source)

        # Load tokenizer from the saved checkpoint directory
        logger.info("Loading tokenizer from %s", model_source)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)

        # Reconstruct the EXACT architecture used during training
        # (ContractRiskClassifier: AutoModel base + Dropout + Linear on [CLS])
        # then load the fine-tuned weights from model.safetensors.
        logger.info("Reconstructing ContractRiskClassifier: %s, labels=%d, dropout=%s",
                     model_name, num_labels, dropout)
        self.model = ContractRiskClassifier(model_name, num_labels, dropout)

        model_weights_path = os.path.join(model_source, "model.safetensors")
        state = load_file(model_weights_path, device=str(self.device))
        result = self.model.load_state_dict(state)
        logger.info("State dict loaded: %s", result)

        self.model.to(self.device)
        self.model.eval()
        logger.info("RiskPredictor ready on device: %s", self.device)

    def predict(self, text: str) -> Dict:
        """Run risk prediction on a text clause.

        For long texts, chunk_text splits via sliding window. The chunk
        with the *highest* predicted risk label wins (max-risk policy:
        a genuine high-risk signal in one chunk should not be diluted
        by averaging with other chunks).

        Returns:
            Dict with ``risk_level`` (str), ``risk_label`` (int),
            ``confidence`` (float — real softmax probability).
        """
        text = clean_text(text)
        if not text:
            return {"risk_level": "low", "risk_label": 0, "confidence": 0.0}

        # Reserve 2 tokens for [CLS] and [SEP]
        effective_max_tokens = self.max_length - 2
        chunks = chunk_text(
            text, self.tokenizer,
            max_tokens=effective_max_tokens, stride=self.stride,
        )

        best_label = -1
        best_confidence = 0.0

        with torch.no_grad():
            for chunk in chunks:
                encoded = self.tokenizer(
                    chunk,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                token_type_ids = encoded.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
                pred_label = int(torch.argmax(probs).item())
                pred_confidence = float(probs[pred_label].item())

                # Max-risk policy: higher label index = higher risk
                if pred_label > best_label or (
                    pred_label == best_label and pred_confidence > best_confidence
                ):
                    best_label = pred_label
                    best_confidence = pred_confidence

        return {
            "risk_level": LABEL_NAMES[best_label],
            "risk_label": best_label,
            "confidence": round(best_confidence, 4),
        }

    @staticmethod
    def get_risk_reason(risk_label: int) -> str:
        """Return a human-readable heuristic explanation for the risk level.

        IMPORTANT: This is a STATIC HEURISTIC LOOKUP based on the
        training data's risk category definitions, NOT model-generated
        explainability. The model does not explain its own decisions.
        This text simply describes what kinds of clauses typically
        fall into each risk tier, as defined during data labeling.
        """
        if risk_label == 2:
            examples = ", ".join(sorted(HIGH_RISK_CLAUSES)[:5])
            return (
                f"Classified as HIGH risk. Clauses in this category typically involve "
                f"restrictive or liability-heavy provisions such as: {examples}. "
                f"(Note: this is a heuristic category description, not a model-generated explanation.)"
            )
        elif risk_label == 1:
            examples = ", ".join(sorted(MEDIUM_RISK_CLAUSES)[:5])
            return (
                f"Classified as MEDIUM risk. Clauses in this category typically involve "
                f"operational or compliance provisions such as: {examples}. "
                f"(Note: this is a heuristic category description, not a model-generated explanation.)"
            )
        else:
            return (
                "Classified as LOW risk. Clauses in this category typically involve "
                "standard administrative or informational provisions. "
                "(Note: this is a heuristic category description, not a model-generated explanation.)"
            )
