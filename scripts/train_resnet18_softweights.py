import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE, IMAGE_DIR, NUM_WORKERS, PROCESSED_DIR, RANDOM_SEED, RESNET18_CONFIG, CLASS_NAMES
)
from src.dataset import SkinLesionDataset, compute_class_weights
from src.model import build_resnet18
from src.train import set_seed, train
from src.transforms import get_train_transform, get_val_test_transform
from src.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def smoke_test(model, train_loader, class_weights, device):
    logger.info("--- Running 10-batch Smoke Test ---")
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 10:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        logger.info(f"Smoke batch {batch_idx + 1}/10 - Loss: {loss.item():.4f}")
        
    logger.info("Smoke test passed! Forward, backward, and optimizer steps are working.")
    logger.info("----------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 with softened class weights.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of epochs to train for.")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    
    class_weights = compute_class_weights(train_df, soften=True)
    logger.info("EXACT NEW CLASS WEIGHTS (Softened):")
    for k, v in zip(CLASS_NAMES, class_weights.tolist()):
        logger.info(f"{k}: {v:.4f}")
        
    train_transform = get_train_transform()
    val_transform = get_val_test_transform()

    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Note: re-initialize model so smoke test updates don't carry over into the fresh start
    if not args.resume:
        smoke_model = build_resnet18(num_classes=7, pretrained=True, freeze_backbone=True).to(device)
        smoke_test(smoke_model, train_loader, class_weights, device)
        del smoke_model
    
    # Fresh model for actual training
    set_seed(RANDOM_SEED)
    model = build_resnet18(num_classes=7, pretrained=True, freeze_backbone=True)
    
    config = dict(RESNET18_CONFIG)
    config["model_type"] = "resnet18_softweights"
    config["checkpoint_name"] = "skin_lesion_resnet18_softweights.pth"
    config["num_epochs"] = args.epochs
    
    logger.info(f"Starting Full Training Phase 1 (Frozen Backbone). Max Epochs: {args.epochs}")
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config,
        device=device,
        resume_checkpoint=args.resume,
    )
    
    logger.info("Training complete. Best val_f1: %.4f", max(history["val_f1"]))

if __name__ == "__main__":
    main()
