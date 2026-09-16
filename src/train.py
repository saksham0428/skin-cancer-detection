"""
train.py — Training and validation loop for skin lesion classification.

Features:
  - Epoch-level training and validation
  - Class-weighted CrossEntropyLoss (handles class imbalance)
  - Early stopping on validation F1 (not accuracy — imbalanced dataset)
  - Best-checkpoint saving (val F1)
  - Training history saved as JSON
  - Fully reproducible with fixed random seed
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import (
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # No CUDA — torch.cuda.manual_seed_all not needed


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            logger.debug(
                "  batch %d/%d | loss: %.4f",
                batch_idx,
                len(loader),
                loss.item(),
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run validation.

    Returns:
        (avg_loss, accuracy, macro_f1)

    We track macro F1 as the primary validation metric because the dataset is
    heavily imbalanced — accuracy alone would be misleading (a model predicting
    only 'nv' would score ~66% accuracy on train, but F1 near 0 for rare classes).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    config: dict[str, Any],
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Full training loop with early stopping and checkpoint saving.

    Args:
        model:          The model to train.
        train_loader:   DataLoader for the training split.
        val_loader:     DataLoader for the validation split.
        class_weights:  Inverse-frequency weights for CrossEntropyLoss.
        config:         Training configuration dict (from src.config).
        device:         torch.device; defaults to cpu.

    Returns:
        history dict with per-epoch metrics.
    """
    if device is None:
        device = torch.device("cpu")

    set_seed(RANDOM_SEED)
    model = model.to(device)
    class_weights = class_weights.to(device)

    # --- Loss: class-weighted CE to handle imbalance ---
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimiser: Adam with weight decay for regularisation ---
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # --- LR scheduler: reduce on plateau (val F1 stagnation) ---
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",       # maximise val F1
        factor=0.5,
        patience=3,
    )

    model_type = config["model_type"]
    checkpoint_name = config["checkpoint_name"]
    checkpoint_path = MODELS_DIR / checkpoint_name
    patience = config.get("early_stopping_patience", 5)
    num_epochs = config["num_epochs"]

    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_f1 = 0.0
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("Training: %s | device: %s | epochs: %d", model_type, device, num_epochs)
    logger.info("LR: %.5f | weight_decay: %.5f | patience: %d",
                config["learning_rate"], config.get("weight_decay", 1e-4), patience)
    logger.info("=" * 60)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["val_f1"].append(round(val_f1, 6))
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %02d/%02d | %.0fs | "
            "train_loss=%.4f acc=%.4f | "
            "val_loss=%.4f acc=%.4f f1=%.4f | lr=%.6f",
            epoch, num_epochs, epoch_time,
            train_loss, train_acc,
            val_loss, val_acc, val_f1,
            current_lr,
        )

        # Checkpoint on improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "preprocessing": {
                        "image_size": 224,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                    "epoch": epoch,
                    "val_f1": round(best_val_f1, 6),
                    "val_acc": round(val_acc, 6),
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info("  ✓ Checkpoint saved (val_f1=%.4f)", best_val_f1)
        else:
            patience_counter += 1
            logger.info(
                "  No improvement (%d/%d patience)", patience_counter, patience
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, best_val_f1)
            break

    # Save training history as JSON for reproducibility/plotting
    history_path = MODELS_DIR / f"training_history_{model_type}.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "model_type": model_type,
                "best_val_f1": round(best_val_f1, 6),
                "history": history,
                "config": config,
            },
            f,
            indent=2,
        )
    logger.info("Training history saved: %s", history_path)
    logger.info("Best checkpoint: %s | val_f1=%.4f", checkpoint_path, best_val_f1)

    return history
