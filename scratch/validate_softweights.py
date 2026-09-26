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

    model = build_resnet18(num_classes=7, pretrained=False)
    
    ckpt_path = PROJECT_ROOT / "models" / "skin_lesion_resnet18_softweights.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_max_confs = []
    image_ids = val_df['image_id'].tolist() if 'image_id' in val_df.columns else [f"img_{i}" for i in range(len(val_df))]
    
    # Check if 'lesion_id' exists
    has_lesion_id = 'lesion_id' in val_df.columns
    lesion_ids = val_df['lesion_id'].tolist() if has_lesion_id else [None]*len(val_df)

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
    # 2. Macro precision
    macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    # 3. Macro recall
    macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    # 4. Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print("\n--- OVERALL METRICS ---")
    print(f"1. Validation Accuracy: {acc:.4f}")
    print(f"2. Macro Precision: {macro_p:.4f}")
    print(f"3. Macro Recall: {macro_r:.4f}")
    print(f"4. Macro F1: {macro_f1:.4f}")

    # 5. Per-class metrics
    print("\n--- 5. PER-CLASS METRICS ---")
    report_dict = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    # 6. Prediction counts
    print("\n--- 6. PREDICTION COUNTS ---")
    pred_counts = pd.Series(all_preds).value_counts().sort_index()
    for i, count in pred_counts.items():
        print(f"{CLASS_NAMES[i]}: {count}")

    cm = confusion_matrix(all_labels, all_preds)
    def get_cm_val(true_name, pred_name):
        t = CLASS_NAMES.index(true_name)
        p = CLASS_NAMES.index(pred_name)
        return cm[t, p]

    print("\n--- SPECIFIC MISCLASSIFICATIONS ---")
    print(f"7. nv -> mel count: {get_cm_val('nv', 'mel')}")
    print(f"8. mel -> nv count: {get_cm_val('mel', 'nv')}")
    print(f"9. nv -> bkl count: {get_cm_val('nv', 'bkl')}")
    print(f"10. bkl -> nv count: {get_cm_val('bkl', 'nv')}")
    print(f"11. bkl -> mel count: {get_cm_val('bkl', 'mel')}")

    print("\n--- CONFIDENCE METRICS ---")
    # 12. Mean and median max confidence
    mean_conf = np.mean(all_max_confs)
    median_conf = np.median(all_max_confs)
    print(f"12. Mean max confidence: {mean_conf:.4f}")
    print(f"    Median max confidence: {median_conf:.4f}")

    # 13. Low-confidence predictions (<0.5)
    low_conf_count = sum(c < 0.5 for c in all_max_confs)
    print(f"13. Low-confidence predictions (<0.5): {low_conf_count} / {len(all_max_confs)} ({low_conf_count/len(all_max_confs)*100:.2f}%)")

    # 14. Save validation predictions CSV
    preds_df = pd.DataFrame({
        'image_id': image_ids,
        'true_label': [CLASS_NAMES[i] for i in all_labels],
        'pred_label': [CLASS_NAMES[i] for i in all_preds],
        'max_confidence': all_max_confs,
    })
    if has_lesion_id:
        preds_df['lesion_id'] = lesion_ids
        
    for i, c in enumerate(CLASS_NAMES):
        preds_df[f'prob_{c}'] = [p[i] for p in all_probs]
        
    csv_path = PROJECT_ROOT / "models" / "val_predictions_resnet18_softweights.csv"
    preds_df.to_csv(csv_path, index=False)
    print(f"\n14. Saved validation predictions to {csv_path}")

    # 15. Save confusion matrices
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Validation Confusion Matrix (Raw)')
    raw_cm_path = PROJECT_ROOT / "models" / "val_cm_raw_resnet18_softweights.png"
    plt.savefig(raw_cm_path)
    plt.close()

    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Validation Confusion Matrix (Normalized by True)')
    norm_cm_path = PROJECT_ROOT / "models" / "val_cm_norm_resnet18_softweights.png"
    plt.savefig(norm_cm_path)
    plt.close()

    print(f"15. Saved raw confusion matrix to {raw_cm_path}")
    print(f"    Saved normalized confusion matrix to {norm_cm_path}")

if __name__ == "__main__":
    main()
