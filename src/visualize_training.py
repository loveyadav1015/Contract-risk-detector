"""Training visualization and evaluation reporting.

This module provides:
- plot_training_curves: Plots loss, F1, and other metrics from
  HuggingFace Trainer's log_history.
- plot_confusion_matrix: Runs inference on a test set and produces
  a confusion matrix heatmap plus sklearn classification_report.
- save_training_history: Persists raw log_history as JSON for
  auditability.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window needed)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.utils import get_logger

logger = get_logger(__name__)

LABEL_NAMES = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def save_training_history(
    log_history: List[Dict],
    output_dir: str = "models",
) -> str:
    """Persist raw log_history as JSON for auditability.

    Returns:
        Absolute path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_history.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(log_history, fh, indent=2, default=str)
    logger.info("Training history saved to %s", path)
    return os.path.abspath(path)


def plot_training_curves(
    log_history: List[Dict],
    output_dir: str = "models",
) -> Optional[str]:
    """Plot training & evaluation curves from Trainer's log_history.

    Creates a 2×2 figure:
        (a) Training loss vs. step
        (b) Eval loss vs. epoch
        (c) Eval F1 macro vs. epoch
        (d) Eval accuracy, precision_macro, recall_macro vs. epoch

    If the required keys are missing from *log_history*, the
    corresponding subplot is left blank with a warning annotation
    rather than fabricated data.

    Returns:
        Absolute path to the saved PNG, or ``None`` if log_history
        was empty / unusable.
    """
    if not log_history:
        print("WARNING: log_history is empty — cannot plot training curves.")
        return None

    # Separate training-step entries from eval-epoch entries
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries = [e for e in log_history if "eval_loss" in e]

    if not train_entries and not eval_entries:
        print("WARNING: log_history has no loss or eval entries — skipping curves.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training & Evaluation Curves", fontsize=14, fontweight="bold")

    # --- (a) Training loss vs step ---
    ax = axes[0, 0]
    if train_entries:
        steps = [e.get("step", i) for i, e in enumerate(train_entries)]
        losses = [e["loss"] for e in train_entries]
        ax.plot(steps, losses, color="steelblue", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("(a) Training Loss vs. Step")
    else:
        ax.text(0.5, 0.5, "No training loss data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("(a) Training Loss — unavailable")

    # --- (b) Eval loss vs epoch ---
    ax = axes[0, 1]
    if eval_entries and all("eval_loss" in e for e in eval_entries):
        epochs = [e.get("epoch", i + 1) for i, e in enumerate(eval_entries)]
        eval_losses = [e["eval_loss"] for e in eval_entries]
        ax.plot(epochs, eval_losses, color="indianred", marker="o", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("(b) Eval Loss vs. Epoch")
    else:
        ax.text(0.5, 0.5, "No eval loss data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("(b) Eval Loss — unavailable")

    # --- (c) Eval F1 macro vs epoch ---
    ax = axes[1, 0]
    if eval_entries and all("eval_f1_macro" in e for e in eval_entries):
        epochs = [e.get("epoch", i + 1) for i, e in enumerate(eval_entries)]
        f1s = [e["eval_f1_macro"] for e in eval_entries]
        ax.plot(epochs, f1s, color="forestgreen", marker="s", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 (macro)")
        ax.set_title("(c) Eval F1 Macro vs. Epoch")
    else:
        ax.text(0.5, 0.5, "No eval F1 data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("(c) Eval F1 — unavailable")

    # --- (d) Eval accuracy, precision, recall vs epoch ---
    ax = axes[1, 1]
    has_all = (
        eval_entries
        and all("eval_accuracy" in e for e in eval_entries)
        and all("eval_precision_macro" in e for e in eval_entries)
        and all("eval_recall_macro" in e for e in eval_entries)
    )
    if has_all:
        epochs = [e.get("epoch", i + 1) for i, e in enumerate(eval_entries)]
        ax.plot(epochs, [e["eval_accuracy"] for e in eval_entries],
                marker="o", label="Accuracy", color="steelblue")
        ax.plot(epochs, [e["eval_precision_macro"] for e in eval_entries],
                marker="^", label="Precision (macro)", color="darkorange")
        ax.plot(epochs, [e["eval_recall_macro"] for e in eval_entries],
                marker="v", label="Recall (macro)", color="mediumorchid")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("(d) Eval Metrics vs. Epoch")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Missing eval metric keys", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title("(d) Eval Metrics — unavailable")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", save_path)
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    model: torch.nn.Module,
    test_dataset,
    output_dir: str = "models",
    batch_size: int = 8,
) -> str:
    """Run inference on *test_dataset*, produce a confusion matrix
    heatmap, and print the full ``classification_report``.

    Args:
        model: A trained ContractRiskClassifier (already on device).
        test_dataset: A ContractClauseDataset (the test split).
        output_dir: Directory to save the confusion matrix PNG.
        batch_size: Batch size for inference.

    Returns:
        Absolute path to the saved confusion matrix PNG.
    """
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # --- Classification report (printed to console) ---
    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_NAMES,
        labels=[0, 1, 2],
        zero_division=0,
    )
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print("Raw confusion matrix:")
    print(cm)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)
    return os.path.abspath(save_path)
