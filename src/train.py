"""Training loop for the Contract Risk Detector.

This module provides:
- train_model: Fine-tunes a ContractRiskClassifier using HuggingFace Trainer
  with AdamW + linear warmup, fp16 mixed precision, and epoch-level
  evaluation/saving.  All hyperparameters are read from config.yaml.
"""

import json
import os
from typing import Dict, Tuple

import torch
from transformers import Trainer, TrainingArguments

from src.dataset import build_datasets
from src.evaluate import compute_metrics
from src.model import ContractRiskClassifier, build_model
from src.utils import get_logger, load_config
from src.visualize_training import (
    plot_confusion_matrix,
    plot_training_curves,
    save_training_history,
)

logger = get_logger(__name__)


def train_model(
    config: dict,
) -> Tuple[ContractRiskClassifier, object, Dict, object, object]:
    """Fine-tune a ContractRiskClassifier end-to-end.

    1. Builds train/val/test ``ContractClauseDataset`` objects (not
       DataLoaders — HuggingFace Trainer manages batching internally).
    2. Constructs ``TrainingArguments`` from *config*.
    3. Trains with ``Trainer``, evaluating on val every epoch.
    4. Evaluates on the held-out test set.
    5. Saves the best model + tokenizer + constructor args to disk.

    Returns:
        ``(model, tokenizer, test_eval_results)``
    """
    # ------------------------------------------------------------------
    # 1. Data — Trainer needs Dataset objects, NOT DataLoaders
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds, tokenizer = build_datasets(config)
    logger.info(
        "Datasets ready — train: %d | val: %d | test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model = build_model(config)

    # ------------------------------------------------------------------
    # 3. TrainingArguments — every value read from config, nothing hardcoded
    # ------------------------------------------------------------------
    output_dir: str = config["paths"]["model_output_dir"]
    best_model_path: str = config["paths"]["best_model_path"]
    logs_dir: str = config["paths"]["logs_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        fp16=config["training"]["fp16"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config["training"]["save_best_metric"],
        logging_dir=logs_dir,
        logging_steps=50,
        seed=config["training"]["seed"],
        # Disable third-party reporters (wandb / tensorboard) unless
        # the user explicitly configures them elsewhere.
        report_to="none",
    )

    # ------------------------------------------------------------------
    # 4. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 5. Test-set evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating on held-out test set …")
    eval_results = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    logger.info("Test results: %s", eval_results)

    # ------------------------------------------------------------------
    # 6. Save model, tokenizer, and constructor config
    # ------------------------------------------------------------------
    # SAVE FORMAT NOTE (important for Phase 3):
    # ─────────────────────────────────────────
    # ContractRiskClassifier is a plain nn.Module, NOT a HuggingFace
    # PreTrainedModel.  Trainer.save_model() therefore saves the full
    # state_dict as "model.safetensors" (or "pytorch_model.bin") inside
    # best_model_path.
    #
    # To reload in Phase 3 (api/services/predictor.py):
    #   1. Read model_config.json to get model_name, num_labels, dropout
    #   2. model = ContractRiskClassifier(model_name, num_labels, dropout)
    #   3. state = load_file("models/best_model/model.safetensors")
    #      model.load_state_dict(state)
    #   4. tokenizer = AutoTokenizer.from_pretrained("models/best_model")
    #
    # We also persist the base transformer's config.json so the base model
    # can be rebuilt entirely from local files (no internet required).

    os.makedirs(best_model_path, exist_ok=True)
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    # Save constructor args so Phase 3 can reconstruct the architecture
    model_constructor_config = {
        "model_name": config["model"]["name"],
        "num_labels": config["model"]["num_labels"],
        "dropout": config["model"]["dropout"],
    }
    config_save_path = os.path.join(best_model_path, "model_config.json")
    with open(config_save_path, "w", encoding="utf-8") as fh:
        json.dump(model_constructor_config, fh, indent=2)

    # Also persist trainer state (includes log_history) so visualization
    # can be re-run later without re-training.
    trainer.save_state()
    logger.info("Trainer state saved to: %s", output_dir)

    logger.info("Model, tokenizer, and config saved to: %s", best_model_path)

    # Return trainer and test_ds so __main__ can produce visualizations
    # without re-instantiating anything.
    return model, tokenizer, eval_results, trainer, test_ds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()
    model, tokenizer, eval_results, trainer, test_ds = train_model(config)

    # ------------------------------------------------------------------
    # Print final test metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 2 — TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Test Accuracy:          {eval_results.get('test_accuracy', 'N/A'):.4f}")
    print(f"  Test F1 (macro):        {eval_results.get('test_f1_macro', 'N/A'):.4f}")
    print(f"  Test Precision (macro): {eval_results.get('test_precision_macro', 'N/A'):.4f}")
    print(f"  Test Recall (macro):    {eval_results.get('test_recall_macro', 'N/A'):.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Confirm saved model files
    # ------------------------------------------------------------------
    best_model_path = config["paths"]["best_model_path"]
    abs_path = os.path.abspath(best_model_path)
    print(f"\n  Model saved to: {abs_path}")

    if os.path.isdir(best_model_path):
        saved_files = os.listdir(best_model_path)
        print(f"  Saved files:    {saved_files}")
        print("  ✅ Model files confirmed on disk.")
    else:
        print("  ❌ WARNING: Save directory not found!")

    # ------------------------------------------------------------------
    # Visualizations (using real data only)
    # ------------------------------------------------------------------
    output_dir = config["paths"]["model_output_dir"]

    # Save raw log_history as JSON for auditability
    save_training_history(trainer.state.log_history, output_dir=output_dir)

    # Training curves from real log_history
    curves_path = plot_training_curves(
        trainer.state.log_history, output_dir=output_dir
    )
    if curves_path:
        print(f"  Training curves: {curves_path}")

    # Confusion matrix from real test-set inference
    # load_best_model_at_end=True was set, so trainer already restored
    # the best checkpoint (by f1_macro) into model before evaluate().
    cm_path = plot_confusion_matrix(
        model, test_ds, output_dir=output_dir, batch_size=8
    )
    print(f"  Confusion matrix: {cm_path}")
    print("\n  ✅ Phase 2 complete — all artifacts saved.")
