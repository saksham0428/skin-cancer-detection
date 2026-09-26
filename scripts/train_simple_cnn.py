"""
train_simple_cnn.py — End-to-end training script for the SimpleCNN baseline.

Run from the project root:
    .venv\\Scripts\\python.exe scripts/train_simple_cnn.py

The best checkpoint (by validation macro F1) is saved to:
    models/skin_lesion_cnn_baseline.pth

Training history is saved to:
    models/training_history_simple_cnn.json
"""

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
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
    SIMPLE_CNN_CONFIG,
)
from src.dataset import SkinLesionDataset, compute_class_weights
from src.model import SimpleCNN, count_trainable_params, count_total_params
from src.train import set_seed, train
from src.transforms import get_train_transform, get_val_test_transform

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    set_seed(RANDOM_SEED)
    from src.device import get_device
    device = get_device()

    # --- Load splits ---
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    # --- Class weights (imbalance handling) ---
    class_weights = compute_class_weights(train_df)
    logger.info("Class weights: %s", {k: f"{v:.2f}" for k, v in zip(
        ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        class_weights.tolist()
    )})

    # --- Transforms ---
    train_transform = get_train_transform()
    val_transform = get_val_test_transform()

    # --- Datasets ---
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIR, val_transform)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # --- Model ---
    model = SimpleCNN(num_classes=7)
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    logger.info("SimpleCNN | trainable params: %s / %s", f"{trainable:,}", f"{total:,}")

    # --- Train ---
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=SIMPLE_CNN_CONFIG,
        device=device,
    )

    logger.info("Training complete.")
    logger.info("Best val_f1: %.4f", max(history["val_f1"]))


if __name__ == "__main__":
    main()
