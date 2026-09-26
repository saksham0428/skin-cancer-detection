import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path("e:/Project/Saksham/skin-cancer-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_resnet18
from src.train import train
from torch.utils.data import DataLoader, TensorDataset
from src.config import RESNET18_CONFIG
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    device = torch.device("cpu")
    model = build_resnet18(num_classes=7, pretrained=False, freeze_backbone=True)
    
    old_ckpt_path = PROJECT_ROOT / "models" / "skin_lesion_resnet18_baseline.pth"
    
    print(f"Testing old checkpoint: {old_ckpt_path}")
    ckpt = torch.load(old_ckpt_path, map_location="cpu", weights_only=False)
    
    if "optimizer_state_dict" not in ckpt:
        print("PASS: Old checkpoint correctly lacks optimizer state.")
    else:
        print("FAIL: Old checkpoint has optimizer state.")
        
    print("Testing save/load format via torch directly...")
    
    # Mock states
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="max")
    patience_counter = 2
    epoch = 5
    best_val_f1 = 0.75
    best_val_loss = 0.45
    
    new_ckpt_path = PROJECT_ROOT / "models" / "test_resume.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "patience_counter": patience_counter,
            "epoch": epoch,
            "val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
        },
        new_ckpt_path
    )
    
    print("Saved new checkpoint.")
    
    # Reload
    new_ckpt = torch.load(new_ckpt_path, map_location="cpu", weights_only=False)
    if "optimizer_state_dict" in new_ckpt and "scheduler_state_dict" in new_ckpt:
        print("PASS: New checkpoint has optimizer and scheduler state.")
    else:
        print("FAIL: Missing states in new checkpoint.")
        
    print(f"Epoch restored: {new_ckpt['epoch']} (expected 5)")
    print(f"Patience restored: {new_ckpt['patience_counter']} (expected 2)")
    print(f"Best Val F1 restored: {new_ckpt['val_f1']} (expected 0.75)")
    print(f"Best Val Loss restored: {new_ckpt['best_val_loss']} (expected 0.45)")

if __name__ == '__main__':
    main()
