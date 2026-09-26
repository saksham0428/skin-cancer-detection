"""
transforms.py — Image preprocessing pipelines.

Single source of truth for all transform pipelines used during:
  - training  (with augmentation)
  - validation / test  (no augmentation)
  - inference  (same as val/test — consistency with training is critical)

The inference transform is intentionally identical to val_test_transform.
Any change here must be reflected in saved model metadata.
"""

from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform() -> transforms.Compose:
    """
    Augmented transform pipeline for training.

    Augmentations applied:
      - RandomHorizontalFlip(p=0.5)
      - RandomVerticalFlip(p=0.5)
      - RandomRotation(±15°)

    Followed by resize, ToTensor, and ImageNet normalisation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transform() -> transforms.Compose:
    """
    Clean transform pipeline for validation, test, and inference.

    No random augmentation — deterministic output for consistent evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Alias — inference uses exactly the same pipeline as validation/test
# so that the model sees the same distribution it was validated on.
get_inference_transform = get_val_test_transform
