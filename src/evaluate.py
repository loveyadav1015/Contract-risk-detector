"""Evaluation metrics for the Contract Risk Detector.

This module provides:
- compute_metrics: Computes accuracy, macro F1, precision, and recall
  from HuggingFace Trainer's EvalPrediction objects.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from src.utils import get_logger

logger = get_logger(__name__)

LABEL_NAMES = {0: "low", 1: "medium", 2: "high"}


def compute_metrics(eval_pred) -> dict:
    """Compute classification metrics from Trainer's EvalPrediction.

    Args:
        eval_pred: HuggingFace ``EvalPrediction`` with ``.predictions``
            (raw logits) and ``.label_ids`` (ground truth).

    Returns:
        Dict with ``accuracy``, ``f1_macro``, ``precision_macro``,
        ``recall_macro``.
    """
    logits = eval_pred.predictions
    label_ids = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_ids, preds, average="macro", zero_division=0
    )

    # Per-class F1 (force all 3 classes so the array always has 3 entries)
    per_class_f1 = f1_score(
        label_ids, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    for cls_id, score in enumerate(per_class_f1):
        logger.info("  F1 [%s]: %.4f", LABEL_NAMES.get(cls_id, str(cls_id)), score)

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }
