"""
model.py — Model architectures for skin lesion classification.

Two models are available:
  1. SimpleCNN     — baseline CNN trained from scratch
  2. ResNet18      — pretrained ImageNet backbone, fine-tuned on HAM10000

Factory functions return a fully configured model ready for training or inference.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.config import NUM_CLASSES


# ---------------------------------------------------------------------------
# 1. SimpleCNN — Baseline
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Baseline convolutional neural network trained from scratch.

    Architecture:
        Conv2d(3→32)  + ReLU + MaxPool2d(2)   →  [B, 32, 112, 112]
        Conv2d(32→64) + ReLU + MaxPool2d(2)   →  [B, 64,  56,  56]
        Conv2d(64→128)+ ReLU + MaxPool2d(2)   →  [B, 128, 28,  28]
        Flatten                                →  [B, 128*28*28]
        Linear(128*28*28 → 256) + ReLU
        Linear(256 → num_classes)

    Input:  [B, 3, 224, 224]
    Output: [B, num_classes]  (raw logits)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),          # regularisation — not in notebook but helps
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# 2. ResNet18 — Transfer Learning
# ---------------------------------------------------------------------------

def build_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Build a ResNet18 model for transfer learning.

    Strategy when freeze_backbone=True:
      - All layers are frozen (no gradient computation).
      - layer4 (last residual block, ~640K params) is unfrozen.
      - The final fully-connected head is replaced and always trained.

    This trains only ~2.8M / ~11.7M total parameters, which is practical
    on a CPU-only machine while still leveraging ImageNet features.

    Args:
        num_classes:      Number of output classes (default: 7).
        pretrained:       Load ImageNet weights (default: True).
        freeze_backbone:  Freeze all layers except layer4 + fc (default: True).

    Returns:
        Configured ResNet18 nn.Module.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 (last residual block) for task-specific fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the ImageNet head (1000 classes → num_classes)
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    # fc is always trainable (new head, not pretrained)

    return model


# ---------------------------------------------------------------------------
# Utility — count trainable parameters
# ---------------------------------------------------------------------------

def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable (gradient-enabled) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Return the total parameter count."""
    return sum(p.numel() for p in model.parameters())
