"""
evaluate.py — Model evaluation on held-out data.

Produces:
  - Overall accuracy
  - Macro precision, recall, F1
  - Weighted precision, recall, F1
  - Per-class precision, recall, F1, support
  - Confusion matrix (saved as PNG)

Designed to be run once on the TEST split after model selection is complete.
Do NOT use this to tune hyperparameters — that would invalidate the test set.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
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

from src.config import CLASS_NAMES, MODELS_DIR

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
    save_dir: Path | None = None,
) -> dict:
    """
    Run full evaluation and return a structured metrics dict.

    Args:
        model:      Trained model in eval mode.
        loader:     DataLoader for the split to evaluate.
        device:     torch.device.
        split_name: Label used in filenames/logs (e.g. "test", "validation").
        save_dir:   Directory to save confusion matrix PNG. Defaults to MODELS_DIR.

    Returns:
        Nested dict with all metrics.
    """
    if save_dir is None:
        save_dir = MODELS_DIR

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # --- Aggregate metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # --- Per-class metrics ---
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    per_class_metrics: dict[str, dict[str, float]] = {}
    for i, class_name in enumerate(CLASS_NAMES):
        per_class_metrics[class_name] = {
            "precision": round(float(per_class_precision[i]), 4),
            "recall": round(float(per_class_recall[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
        }

    # --- Full sklearn report (for logging) ---
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    _save_confusion_matrix(cm, split_name, save_dir)

    metrics = {
        "split": split_name,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_metrics,
    }

    # --- Log ---
    logger.info("=" * 60)
    logger.info("Evaluation results -- %s split", split_name.upper())
    logger.info("=" * 60)
    logger.info("Accuracy:          %.4f", accuracy)
    logger.info("Macro Precision:   %.4f", macro_precision)
    logger.info("Macro Recall:      %.4f", macro_recall)
    logger.info("Macro F1:          %.4f  (primary metric -- imbalanced dataset)", macro_f1)
    logger.info("Weighted F1:       %.4f", weighted_f1)
    logger.info("\nPer-class report:\n%s", report)

    # Save metrics JSON
    metrics_path = save_dir / f"evaluation_metrics_{split_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    return metrics


def _save_confusion_matrix(
    cm: np.ndarray,
    split_name: str,
    save_dir: Path,
) -> None:
    """Save a normalised confusion matrix as a PNG."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    # Avoid div by zero for empty rows
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalised Confusion Matrix — {split_name.title()} Split")

    # Annotate cells
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    plt.tight_layout()
    out_path = save_dir / f"confusion_matrix_{split_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)
