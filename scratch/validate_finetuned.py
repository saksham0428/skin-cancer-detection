import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, PROCESSED_DIR, IMAGE_DIR, BATCH_SIZE
from src.dataset import SkinLesionDataset
from src.model import build_resnet18
from src.transforms import get_val_test_transform
from src.device import get_device

def main():
    device = get_device()
    print(f"Using device: {device}")

    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    print(f"Validation size: {len(val_df)}")

    val_transform = get_val_test_transform()
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Phase 2 model has an unfrozen backbone
    model = build_resnet18(num_classes=7, pretrained=False, freeze_backbone=False)
    
    ckpt_path = PROJECT_ROOT / "models" / "skin_lesion_resnet18_softweights_finetuned.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    
    if not ckpt_path.exists():
        print("Finetuned checkpoint does not exist. Validation failed.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_max_confs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_confs, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_max_confs.extend(max_confs.cpu().numpy())

    # 1. Validation accuracy
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n--- OVERALL METRICS ---")
    print(f"1. Validation Accuracy: {acc:.4f}")
    print(f"2. Macro Precision: {macro_p:.4f}")
    print(f"3. Macro Recall: {macro_r:.4f}")
    print(f"4. Macro F1: {macro_f1:.4f}")

    print("\n--- 5. PER-CLASS METRICS ---")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- 6. PREDICTION COUNTS ---")
    pred_counts = pd.Series(all_preds).value_counts().sort_index()
    for i, count in pred_counts.items():
        print(f"{CLASS_NAMES[i]}: {count}")

    mean_conf = np.mean(all_max_confs)
    median_conf = np.median(all_max_confs)
    print(f"\nMean max confidence: {mean_conf:.4f}")
    print(f"Median max confidence: {median_conf:.4f}")

    low_conf_count = sum(c < 0.5 for c in all_max_confs)
    print(f"Low-confidence predictions (<0.5): {low_conf_count} / {len(all_max_confs)} ({low_conf_count/len(all_max_confs)*100:.2f}%)")

if __name__ == "__main__":
    main()
