"""
dataset.py — PyTorch Dataset for the HAM10000 skin lesion dataset.

Extracted and productionised from notebooks/02_pytorch_dataset.ipynb.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASS_TO_INDEX, CLASS_NAMES, NUM_CLASSES


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset that loads skin lesion images from disk.

    Args:
        dataframe: DataFrame with at minimum 'image_id' and 'dx' columns.
        image_dir: Directory containing the JPG images.
        transform: torchvision transform pipeline to apply to each image.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path | str,
        transform=None,
    ) -> None:
        # Reset index so iloc is safe regardless of how the df was filtered
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]

        image_id: str = row["image_id"]
        label_name: str = row["dx"]

        image_path = self.image_dir / f"{image_id}.jpg"

        # Always convert to RGB — some images may be RGBA or greyscale
        image = Image.open(image_path).convert("RGB")

        label: int = CLASS_TO_INDEX[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Class weighting utilities
# ---------------------------------------------------------------------------

def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Strategy: weight_i = total_samples / (num_classes * count_i)
    This is the sklearn 'balanced' formula — it ensures minority classes
    receive proportionally larger gradient updates.

    Args:
        train_df: Training split DataFrame with a 'dx' column.

    Returns:
        Float tensor of shape (NUM_CLASSES,), ordered by CLASS_NAMES index.
    """
    counts = train_df["dx"].value_counts()
    total = len(train_df)

    weights = []
    for class_name in CLASS_NAMES:
        count = counts.get(class_name, 1)  # avoid div-by-zero for unseen classes
        weight = total / (NUM_CLASSES * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
