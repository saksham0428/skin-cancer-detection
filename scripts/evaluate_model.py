"""
evaluate_model.py — Run final evaluation on the test split.

This script should be run ONCE after model selection is complete.
It uses the test split that was NOT used for training or hyperparameter tuning.

Usage:
    .venv\\Scripts\\python.exe scripts/evaluate_model.py --model resnet18
    .venv\\Scripts\\python.exe scripts/evaluate_model.py --model simple_cnn
"""

import argparse
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
    MODELS_DIR,
    NUM_WORKERS,
    PROCESSED_DIR,
    RESNET18_CONFIG,
    SIMPLE_CNN_CONFIG,
)
from src.dataset import SkinLesionDataset
from src.evaluate import evaluate_model
from src.inference import load_model
from src.transforms import get_val_test_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")
    parser.add_argument(
        "--model",
        choices=["resnet18", "simple_cnn"],
        default="resnet18",
        help="Which model checkpoint to evaluate (default: resnet18)",
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation"],
        default="test",
        help="Which data split to evaluate on (default: test)",
    )
    args = parser.parse_args()

    from src.device import get_device
    device = get_device()

    # --- Checkpoint ---
    if args.model == "resnet18":
        ckpt_name = RESNET18_CONFIG["checkpoint_name"]
    else:
        ckpt_name = SIMPLE_CNN_CONFIG["checkpoint_name"]
    checkpoint_path = MODELS_DIR / ckpt_name

    # --- Load model ---
    model = load_model(checkpoint_path, device=device)

    # --- Load data ---
    csv_name = "test.csv" if args.split == "test" else "validation.csv"
    df = pd.read_csv(PROCESSED_DIR / csv_name)
    logger.info("Loaded %s: %d images", csv_name, len(df))

    transform = get_val_test_transform()
    dataset = SkinLesionDataset(df, IMAGE_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Evaluate ---
    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        split_name=args.split,
        save_dir=MODELS_DIR,
    )

    logger.info("Done. Metrics saved to: %s", MODELS_DIR)


if __name__ == "__main__":
    main()
