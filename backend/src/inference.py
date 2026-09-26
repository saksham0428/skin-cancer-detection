"""
inference.py — Production inference pipeline.

This module is the single integration point between the trained model
and any consumer (FastAPI backend, CLI, tests).

The preprocessing pipeline is IDENTICAL to the validation/test pipeline
used during training — no preprocessing mismatch is possible because
both import from src.transforms.

Usage:
    from src.inference import load_model, predict_image

    model = load_model("models/skin_lesion_resnet18.pth")
    result = predict_image(image_bytes, model, device)
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

from src.config import (
    CLASS_FULL_NAMES,
    CLASS_NAMES,
    INDEX_TO_CLASS,
    MEDICAL_DISCLAIMER,
    MODELS_DIR,
    NUM_CLASSES,
)
from src.model import SimpleCNN, build_resnet18
from src.transforms import get_inference_transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Structured result returned by the inference pipeline."""
    predicted_class: str
    predicted_class_full_name: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    disclaimer: str = MEDICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path | str | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load a trained model from a .pth checkpoint.

    The checkpoint must contain:
        - 'model_state_dict': model weights
        - 'model_type': 'simple_cnn' or 'resnet18'
        - 'num_classes': int
        - 'class_names': list[str]

    Args:
        checkpoint_path: Path to the .pth file. If None, uses the default
                         checkpoint from src.config.
        device:          torch.device. Defaults to CPU.

    Returns:
        Model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError:      If the checkpoint format is invalid.
    """
    if device is None:
        device = torch.device("cpu")

    if checkpoint_path is None:
        from src.config import DEFAULT_MODEL_CHECKPOINT
        checkpoint_path = MODELS_DIR / DEFAULT_MODEL_CHECKPOINT

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Run a training script first to produce a checkpoint."
        )

    logger.info("Loading checkpoint: %s", checkpoint_path)

    # map_location=device ensures CPU-only machines can load checkpoints
    # even if they were saved on a machine with CUDA.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Validate checkpoint structure
    required_keys = {"model_state_dict", "model_type", "num_classes"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {missing}. "
            "Please retrain and save with the current training script."
        )

    model_type: str = checkpoint["model_type"]
    num_classes: int = checkpoint["num_classes"]

    # Reconstruct the model architecture
    if model_type == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_type in ("resnet18", "resnet18_softweights", "resnet18_softweights_finetuned"):
        # pretrained=False — we load weights from checkpoint, not from ImageNet
        model = build_resnet18(
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,  # all params needed for inference
        )
    else:
        raise RuntimeError(
            f"Unknown model_type '{model_type}' in checkpoint. "
            "Expected 'simple_cnn' or 'resnet18'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Model loaded: %s | classes=%d | val_f1=%.4f (epoch %d)",
        model_type,
        num_classes,
        checkpoint.get("val_f1", float("nan")),
        checkpoint.get("epoch", -1),
    )

    return model


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

_transform = get_inference_transform()


def predict_image(
    image_input: bytes | str | Path,
    model: nn.Module,
    device: torch.device | None = None,
) -> PredictionResult:
    """
    Run the full inference pipeline on one image.

    Pipeline:
        input (bytes | path)
        → open with PIL
        → RGB conversion
        → resize (224×224)
        → ToTensor
        → ImageNet normalisation
        → model forward pass
        → softmax probabilities
        → top-1 prediction

    Args:
        image_input: Raw image bytes, or a file path.
        model:       Loaded model (from load_model()).
        device:      torch.device. Defaults to CPU.

    Returns:
        PredictionResult dataclass.

    Raises:
        ValueError:  If the image cannot be opened or decoded.
        RuntimeError: If inference fails unexpectedly.
    """
    if device is None:
        device = torch.device("cpu")

    # --- 1. Open image ---
    try:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            # bytes from file upload
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot open or decode image: {exc}") from exc

    # --- 2. Preprocess ---
    tensor: torch.Tensor = _transform(image)          # [3, 224, 224]
    tensor = tensor.unsqueeze(0).to(device)            # [1, 3, 224, 224]

    # --- 3. Inference ---
    with torch.no_grad():
        logits: torch.Tensor = model(tensor)           # [1, num_classes]

    # --- 4. Softmax probabilities ---
    probs: torch.Tensor = torch.softmax(logits, dim=1)[0]  # [num_classes]

    # --- 5. Top prediction ---
    class_index: int = int(probs.argmax().item())
    predicted_class: str = INDEX_TO_CLASS[class_index]
    confidence: float = round(float(probs[class_index].item()), 4)

    # Build probability dict — round to 4 decimal places
    probabilities: dict[str, float] = {
        CLASS_NAMES[i]: round(float(probs[i].item()), 4)
        for i in range(len(CLASS_NAMES))
    }

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_class_full_name=CLASS_FULL_NAMES[predicted_class],
        class_index=class_index,
        confidence=confidence,
        probabilities=probabilities,
    )
