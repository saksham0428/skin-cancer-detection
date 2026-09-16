"""
train_resnet18.py — End-to-end fine-tuning script for ResNet18.

Strategy:
  - Load ImageNet-pretrained ResNet18.
  - Freeze all layers except layer4 + the new FC head.
  - Train for up to 20 epochs with early stopping (patience=6 on val macro F1).
  - After the first training phase, optionally unfreeze all layers and
    run a second phase at a very low LR (full fine-tuning).

Run from the project root:
    .venv\\Scripts\\python.exe scripts/train_resnet18.py

The best checkpoint is saved to:
    models/skin_lesion_resnet18.pth
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    IMAGE_DIR,
    NUM_WORKERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RESNET18_CONFIG,
)
from src.dataset import SkinLesionDataset, compute_class_weights
from src.model import build_resnet18, count_trainable_params, count_total_params
from src.train import set_seed, train
from src.transforms import get_train_transform, get_val_test_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    set_seed(RANDOM_SEED)
    device = torch.device("cpu")
    logger.info("Device: %s", device)

    # --- Load splits ---
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    # --- Class weights ---
    class_weights = compute_class_weights(train_df)
    logger.info("Class weights: %s", {k: f"{v:.2f}" for k, v in zip(
        ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        class_weights.tolist()
    )})

    # --- Transforms ---
    train_transform = get_train_transform()
    val_transform = get_val_test_transform()

    # --- Datasets + DataLoaders ---
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIR, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # --- Model: frozen backbone, trainable layer4 + new FC head ---
    model = build_resnet18(num_classes=7, pretrained=True, freeze_backbone=True)
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    logger.info(
        "ResNet18 (phase 1 — layer4+head) | trainable: %s / %s total",
        f"{trainable:,}", f"{total:,}"
    )

    # --- Phase 1: Train frozen backbone ---
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=RESNET18_CONFIG,
        device=device,
    )

    logger.info("Phase 1 complete. Best val_f1: %.4f", max(history["val_f1"]))
    logger.info("Training complete. Checkpoint: models/skin_lesion_resnet18.pth")


if __name__ == "__main__":
    main()
