"""
app/services/model_service.py — Singleton model service for inference.

The model is loaded once at application startup (via FastAPI lifespan).
Subsequent requests reuse the loaded model — no per-request reload.

This prevents the expensive PyTorch model-load from blocking individual
API requests and ensures thread-safe read-only inference.
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.inference import PredictionResult, load_model, predict_image

logger = logging.getLogger(__name__)


class ModelService:
    """
    Singleton that holds the loaded PyTorch model and exposes a predict method.

    Lifecycle:
        1. FastAPI lifespan startup  →  service.load(checkpoint_path, model_type)
        2. Request                   →  service.predict(image_bytes)
        3. FastAPI lifespan shutdown →  service.unload()
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device: torch.device = torch.device("cpu")
        self._model_type: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_type(self) -> str | None:
        return self._model_type

    def load(self, checkpoint_path: Path | str, device: torch.device | None = None) -> None:
        """
        Load the model from a checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint.
            device:          torch.device (defaults to CPU).

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            RuntimeError:      If the checkpoint is malformed.
        """
        if device is not None:
            self._device = device

        logger.info("ModelService: loading checkpoint from %s", checkpoint_path)
        self._model = load_model(checkpoint_path, device=self._device)

        # Extract model_type from the checkpoint metadata
        import torch as _torch
        ckpt = _torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        self._model_type = ckpt.get("model_type", "unknown")

        logger.info("ModelService: model loaded (%s)", self._model_type)

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._model_type = None
        logger.info("ModelService: model unloaded")

    def predict(self, image_bytes: bytes) -> PredictionResult:
        """
        Run inference on raw image bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image.

        Returns:
            PredictionResult dataclass.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError:   If the image cannot be decoded.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. The service is not ready to accept requests."
            )

        return predict_image(image_bytes, self._model, device=self._device)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py and routes
# ---------------------------------------------------------------------------
model_service = ModelService()
