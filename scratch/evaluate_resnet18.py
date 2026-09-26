import json
import torch
import sys
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("e:/Project/Saksham/skin-cancer-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.device import get_device
from src.inference import load_model
from src.dataset import SkinLesionDataset
from src.transforms import get_val_test_transform
from src.config import IMAGE_DIR, PROCESSED_DIR, BATCH_SIZE, NUM_WORKERS, CLASS_NAMES

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint_path = PROJECT_ROOT / "models" / "skin_lesion_resnet18_resume.pth"
    model = load_model(checkpoint_path, device=device)
    model.eval()
    
    test_csv = PROCESSED_DIR / "test.csv"
    df = pd.read_csv(test_csv)
    print(f"Test samples: {len(df)}")
    assert len(df) == 1002, f"Expected 1002 test samples, got {len(df)}"
    
    transform = get_val_test_transform()
    dataset = SkinLesionDataset(df, IMAGE_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    per_class_p = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Support
    support = [all_labels.count(i) for i in range(len(CLASS_NAMES))]
    
    per_class = {}
    for i, c in enumerate(CLASS_NAMES):
        per_class[c] = {
            "precision": per_class_p[i],
            "recall": per_class_r[i],
            "f1": per_class_f1[i],
            "support": support[i]
        }
        
    report = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "per_class": per_class
    }
    
    out_json = PROJECT_ROOT / "models" / "evaluation_resnet18.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Metrics saved to {out_json}")
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix - ResNet18 (Test Split)")
    
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.tight_layout()
    out_png = PROJECT_ROOT / "models" / "confusion_matrix_resnet18.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {out_png}")
    
    print("\nFINAL METRICS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall: {macro_r:.4f}")
    print("\nPER CLASS:")
    for k, v in per_class.items():
        print(f"  {k}: P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f} S={v['support']}")

if __name__ == "__main__":
    main()
