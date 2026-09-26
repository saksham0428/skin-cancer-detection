import json
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("e:/Project/Saksham/skin-cancer-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.device import get_device
from src.inference import load_model
from src.dataset import SkinLesionDataset
from src.transforms import get_val_test_transform
from src.config import IMAGE_DIR, PROCESSED_DIR, BATCH_SIZE, NUM_WORKERS, CLASS_NAMES, CLASS_TO_INDEX, INDEX_TO_CLASS

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint_path = PROJECT_ROOT / "models" / "skin_lesion_resnet18_softweights.pth"
    model = load_model(checkpoint_path, device=device)
    model.eval()
    
    val_csv = PROCESSED_DIR / "validation.csv"
    df = pd.read_csv(val_csv)
    print(f"Validation samples: {len(df)}")
    assert len(df) == 1002, f"Expected 1002 validation samples, got {len(df)}"
    
    transform = get_val_test_transform()
    dataset = SkinLesionDataset(df, IMAGE_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    all_preds = []
    all_labels = []
    all_confs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            confidences, predicted = torch.max(probs, dim=1)
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confidences.cpu().tolist())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    per_class_p = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Support
    support = [all_labels.count(i) for i in range(len(CLASS_NAMES))]
    
    per_class = {}
    for i, c in enumerate(CLASS_NAMES):
        per_class[c] = {
            "precision": round(per_class_p[i], 4),
            "recall": round(per_class_r[i], 4),
            "f1": round(per_class_f1[i], 4),
            "support": support[i]
        }
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Common incorrect predictions per class
    common_errors = {}
    for i, c in enumerate(CLASS_NAMES):
        row = cm[i]
        errors = [(CLASS_NAMES[j], int(row[j])) for j in range(len(CLASS_NAMES)) if j != i and row[j] > 0]
        errors.sort(key=lambda x: x[1], reverse=True)
        common_errors[c] = errors
        
    # Predicted vs True counts
    true_counts = support
    pred_counts = [all_preds.count(i) for i in range(len(CLASS_NAMES))]
    counts_comparison = {c: {"true": true_counts[i], "predicted": pred_counts[i]} for i, c in enumerate(CLASS_NAMES)}
    
    # Confidence statistics
    mean_conf = float(np.mean(all_confs))
    median_conf = float(np.median(all_confs))
    low_conf_thresh = 0.5
    low_conf_count = int(sum(1 for c in all_confs if c < low_conf_thresh))
    
    report = {
        "aggregate": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4)
        },
        "per_class": per_class,
        "counts_comparison": counts_comparison,
        "common_errors": common_errors,
        "confidence_stats": {
            "mean_max_prob": round(mean_conf, 4),
            "median_max_prob": round(median_conf, 4),
            f"low_confidence_count_lt_{low_conf_thresh}": low_conf_count,
            "total_samples": len(all_labels)
        }
    }
    
    out_json = PROJECT_ROOT / "models" / "error_analysis_resnet18_validation.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Saved JSON report to {out_json}")
    
    # Save CSV
    df["predicted_label"] = [INDEX_TO_CLASS[p] for p in all_preds]
    df["confidence"] = [round(c, 4) for c in all_confs]
    df["is_correct"] = [int(p == l) for p, l in zip(all_preds, all_labels)]
    
    # df has image_id, dx, etc. Let's create a clean subset
    csv_df = pd.DataFrame({
        "image_id": df["image_id"],
        "true_label": df["dx"],
        "predicted_label": df["predicted_label"],
        "confidence": df["confidence"],
        "is_correct": df["is_correct"]
    })
    
    out_csv = PROJECT_ROOT / "models" / "validation_predictions.csv"
    csv_df.to_csv(out_csv, index=False)
    print(f"Saved CSV predictions to {out_csv}")
    
    # Plot unnormalized CM
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Validation Split")
    
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm[i, j] > cm.max()/2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
            
    plt.tight_layout()
    out_png_unnorm = PROJECT_ROOT / "models" / "confusion_matrix_resnet18_validation.png"
    plt.savefig(out_png_unnorm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot normalized CM
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
    ax.set_title("Normalized Confusion Matrix - Validation Split")
    
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.tight_layout()
    out_png_norm = PROJECT_ROOT / "models" / "confusion_matrix_resnet18_validation_normalized.png"
    plt.savefig(out_png_norm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrices")
    
    # Top 3 error patterns based on normalized CM (highest off-diagonal values)
    # We will compute the top 3 highest absolute error counts, AND highest normalized.
    # We will just print them all out so the agent can read and interpret.
    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall: {macro_r:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nCounts (True -> Pred):")
    for c in CLASS_NAMES:
        print(f"{c}: True={counts_comparison[c]['true']} Pred={counts_comparison[c]['predicted']}")
    
    print("\nCommon Errors (True class -> mistakenly predicted as):")
    for c in CLASS_NAMES:
        print(f"  {c}: {common_errors[c]}")
        
    print(f"\nConfidence:")
    print(f"Mean Max Prob: {mean_conf:.4f}")
    print(f"Median Max Prob: {median_conf:.4f}")
    print(f"Low Conf Count (<0.5): {low_conf_count}")

if __name__ == "__main__":
    main()
