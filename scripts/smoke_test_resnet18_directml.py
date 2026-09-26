import logging
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE, IMAGE_DIR, NUM_WORKERS, PROCESSED_DIR, RANDOM_SEED
)
from src.dataset import SkinLesionDataset, compute_class_weights
from src.model import build_resnet18
from src.train import set_seed
from src.transforms import get_train_transform, get_val_test_transform
from src.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    logger.info(f"Using device for ResNet18 smoke test: {device}")
    
    if "privateuseone" not in str(device):
        logger.error("Device is not DirectML! Exiting smoke test.")
        return

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    
    class_weights = compute_class_weights(train_df).to(device)
    
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, get_train_transform())
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIR, get_val_test_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 6. use torchvision's ResNet18 architecture, pretrained ImageNet weights
    try:
        model = build_resnet18(num_classes=7, pretrained=True, freeze_backbone=True).to(device)
        logger.info("ResNet18 pretrained weights successfully loaded: YES")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        logger.info("ResNet18 pretrained weights successfully loaded: NO")
        return
        
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    logger.info("Starting ResNet18 training smoke test (max 10 batches)...")
    model.train()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 10:
            break
            
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    logger.info(f"ResNet18 Training smoke test done.")

    logger.info("Starting ResNet18 validation smoke test (max 5 batches)...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= 5:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
    logger.info(f"ResNet18 Validation smoke test done.")
    logger.info("ResNet18 Smoke Test PASS: Forward/Backward/Optimizer completed successfully.")

if __name__ == "__main__":
    main()
