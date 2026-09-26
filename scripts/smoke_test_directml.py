import logging
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BATCH_SIZE, IMAGE_DIR, NUM_WORKERS, PROCESSED_DIR, RANDOM_SEED
)
from src.dataset import SkinLesionDataset, compute_class_weights
from src.model import SimpleCNN
from src.train import set_seed
from src.transforms import get_train_transform, get_val_test_transform
from src.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    logger.info(f"Using device for smoke test: {device}")
    
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

    model = SimpleCNN(num_classes=7).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logger.info("Starting training smoke test (max 100 batches)...")
    model.train()
    
    train_start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 100:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    train_time = time.time() - train_start_time
    train_batches = min(100, len(train_loader))
    sec_per_train_batch = train_time / train_batches if train_batches > 0 else 0
    train_loss = running_loss / total if total > 0 else 0
    train_acc = correct / total if total > 0 else 0
    
    logger.info(f"Training smoke test done: {train_batches} batches in {train_time:.2f}s ({sec_per_train_batch:.4f}s/batch)")
    logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    logger.info("Starting validation smoke test (max 20 batches)...")
    model.eval()
    val_start_time = time.time()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= 20:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    val_time = time.time() - val_start_time
    val_batches = min(20, len(val_loader))
    sec_per_val_batch = val_time / val_batches if val_batches > 0 else 0
    val_loss = val_running_loss / val_total if val_total > 0 else 0
    val_acc = val_correct / val_total if val_total > 0 else 0
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    logger.info(f"Validation smoke test done: {val_batches} batches in {val_time:.2f}s ({sec_per_val_batch:.4f}s/batch)")
    logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    # Estimates
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    est_epoch_time = (total_train_batches * sec_per_train_batch) + (total_val_batches * sec_per_val_batch)
    est_15_epochs = est_epoch_time * 15
    
    logger.info("=== SMOKE TEST METRICS ===")
    logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    logger.info(f"Seconds per train batch: {sec_per_train_batch:.4f}s")
    logger.info(f"Seconds per val batch: {sec_per_val_batch:.4f}s")
    logger.info(f"Estimated full-epoch time: {est_epoch_time:.2f}s")
    logger.info(f"Estimated 15-epoch time: {est_15_epochs:.2f}s ({est_15_epochs/60:.2f} minutes)")

if __name__ == "__main__":
    main()
