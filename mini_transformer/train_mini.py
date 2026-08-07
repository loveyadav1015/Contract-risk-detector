"""Manual PyTorch training loop for the mini-transformer classifier."""

import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from mini_transformer.attention import create_padding_mask
from mini_transformer.dataset import MiniTransformerDataset, load_splits_from_existing_pipeline
from mini_transformer.model import MiniTransformerClassifier
from mini_transformer.tokenizer import Vocabulary
from src.utils import get_logger, load_config

logger = get_logger(__name__)
LABEL_NAMES: List[str] = ["low", "medium", "high"]


def _extract_label_names(config: Dict) -> List[str]:
    id2label = config.get("labels", {}).get("id2label", {})
    if not id2label:
        return ["low", "medium", "high"]
    sorted_items = sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])
    return [value for _, value in sorted_items]


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    pad_token_id: int,
) -> Tuple[float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            labels = batch["label"].to(device)
            mask = create_padding_mask(token_ids, pad_token_id=pad_token_id).to(device)

            logits = model(token_ids, mask=mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, all_labels, all_preds


def train_mini_transformer(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    d_model: int = 256,
    num_heads: int = 4,
    d_ff: int = 1024,
    num_layers: int = 4,
    max_len: int = 64,
    dropout: float = 0.1,
    output_dir: str = "mini_transformer/best_model_v2",
) -> Dict:
    """Train mini-transformer classifier with a manual PyTorch loop."""
    global LABEL_NAMES

    config = load_config()
    LABEL_NAMES = _extract_label_names(config)

    vocab = Vocabulary()
    vocab.load("mini_transformer/vocab.json")

    train_records, val_records, test_records = load_splits_from_existing_pipeline(config)

    label_counts = Counter(record["risk_label"] for record in train_records)
    total = sum(label_counts.values())
    num_classes = len(LABEL_NAMES)
    class_weights = torch.tensor(
        [total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    )

    train_ds = MiniTransformerDataset(train_records, vocab, max_length=max_len)
    val_ds = MiniTransformerDataset(val_records, vocab, max_length=max_len)
    test_ds = MiniTransformerDataset(test_records, vocab, max_length=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    class_weights = class_weights.to(device)
    print(f"Computed class weights from train split counts {dict(label_counts)}: {class_weights.detach().cpu().tolist()}")

    model = MiniTransformerClassifier(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        num_labels=len(LABEL_NAMES),
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "mini_transformer_best.pt")
    best_val_f1 = float("-inf")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision_macro": [],
        "val_recall_macro": [],
        "val_f1_macro": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [Train]",
            unit="batch",
        )
        for batch in progress:
            token_ids = batch["token_ids"].to(device)
            labels = batch["label"].to(device)
            mask = create_padding_mask(token_ids, pad_token_id=vocab.PAD_ID).to(device)

            optimizer.zero_grad()
            logits = model(token_ids, mask=mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        val_loss, val_labels, val_preds = _evaluate_loader(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            pad_token_id=vocab.PAD_ID,
        )

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(
            val_labels, val_preds, average="macro", zero_division=0
        )
        val_recall = recall_score(
            val_labels, val_preds, average="macro", zero_division=0
        )
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision_macro"].append(val_precision)
        history["val_recall_macro"].append(val_recall)
        history["val_f1_macro"].append(val_f1)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | val_f1={val_f1:.4f} | "
            f"val_precision={val_precision:.4f} | val_recall={val_recall:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved new best model (val_f1_macro=%.4f) to %s", val_f1, best_model_path)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    logger.info("Reloaded best checkpoint from %s", best_model_path)

    test_loss, test_labels, test_preds = _evaluate_loader(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        pad_token_id=vocab.PAD_ID,
    )

    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(
        test_labels, test_preds, average="macro", zero_division=0
    )
    test_recall = recall_score(
        test_labels, test_preds, average="macro", zero_division=0
    )
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(LABEL_NAMES))))

    vocab.save(os.path.join(output_dir, "vocab.json"))

    final_metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision_macro": test_precision,
        "test_recall_macro": test_recall,
        "test_f1_macro": test_f1,
        "test_confusion_matrix": test_cm.tolist(),
    }
    print("Final test metrics:", final_metrics)

    return {
        **final_metrics,
        "history": history,
        "model": model,
        "test_loader": test_loader,
        "device": str(device),
        "output_dir": output_dir,
    }


def plot_mini_transformer_curves(
    history: Dict,
    output_dir: str = "mini_transformer",
    filename: str = "training_curves_v2.png",
) -> None:
    """Plot epoch-level train/val curves from real logged history."""
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        raise ValueError("History is empty; cannot plot training curves.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mini-Transformer Training & Validation Curves", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, history["train_loss"], marker="o", color="steelblue")
    axes[0, 0].set_title("(a) Train Loss vs. Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epochs, history["val_loss"], marker="o", color="indianred")
    axes[0, 1].set_title("(b) Val Loss vs. Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(epochs, history["val_f1_macro"], marker="s", color="forestgreen")
    axes[1, 0].set_title("(c) Val F1 Macro vs. Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 (macro)")

    axes[1, 1].plot(epochs, history["val_accuracy"], marker="o", label="Accuracy", color="steelblue")
    axes[1, 1].plot(
        epochs,
        history["val_precision_macro"],
        marker="^",
        label="Precision (macro)",
        color="darkorange",
    )
    axes[1, 1].plot(
        epochs,
        history["val_recall_macro"],
        marker="v",
        label="Recall (macro)",
        color="mediumorchid",
    )
    axes[1, 1].set_title("(d) Val Metrics vs. Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", save_path)


def plot_confusion_matrix_mini(
    model,
    test_loader,
    output_dir: str = "mini_transformer",
    filename: str = "confusion_matrix_v2.png",
) -> None:
    """Run real evaluation and save a confusion matrix heatmap."""
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            token_ids = batch["token_ids"].to(device)
            labels = batch["label"].to(device)
            mask = create_padding_mask(token_ids, pad_token_id=0).to(device)

            logits = model(token_ids, mask=mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(LABEL_NAMES))),
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(LABEL_NAMES))))
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
    ax.set_title("Confusion Matrix — Mini-Transformer Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


if __name__ == "__main__":
    run_start = time.time()
    results = train_mini_transformer()
    plot_mini_transformer_curves(results["history"], output_dir="mini_transformer")
    plot_confusion_matrix_mini(results["model"], results["test_loader"], output_dir="mini_transformer")

    print("\n" + "=" * 60)
    print("  PHASE 5 — MINI-TRANSFORMER TEST METRICS")
    print("=" * 60)
    print(f"  Test Loss:              {results['test_loss']:.4f}")
    print(f"  Test Accuracy:          {results['test_accuracy']:.4f}")
    print(f"  Test F1 (macro):        {results['test_f1_macro']:.4f}")
    print(f"  Test Precision (macro): {results['test_precision_macro']:.4f}")
    print(f"  Test Recall (macro):    {results['test_recall_macro']:.4f}")
    print("=" * 60)

    elapsed_sec = time.time() - run_start
    print(f"Estimated total training+evaluation time on this machine: {elapsed_sec / 60.0:.2f} minutes")
    print("✅ Phase 5 (mini-transformer) training complete.")
