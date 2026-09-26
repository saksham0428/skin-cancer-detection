import sys
import json
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

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"Test size: {len(test_df)}")

    test_transform = get_val_test_transform()
    test_dataset = SkinLesionDataset(test_df, IMAGE_DIR, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Phase 2 model
    model = build_resnet18(num_classes=7, pretrained=False, freeze_backbone=False)
    
    ckpt_path = PROJECT_ROOT / "models" / "skin_lesion_resnet18_softweights_finetuned.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_max_confs = []
    image_ids = test_df['image_id'].tolist() if 'image_id' in test_df.columns else [f"img_{i}" for i in range(len(test_df))]
    
    has_lesion_id = 'lesion_id' in test_df.columns
    lesion_ids = test_df['lesion_id'].tolist() if has_lesion_id else [None]*len(test_df)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_confs, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_max_confs.extend(max_confs.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print("\n--- OVERALL TEST METRICS ---")
    print(f"1. Test Accuracy: {acc:.4f}")
    print(f"2. Macro Precision: {macro_p:.4f}")
    print(f"3. Macro Recall: {macro_r:.4f}")
    print(f"4. Macro F1: {macro_f1:.4f}")
    print(f"5. Weighted F1: {weighted_f1:.4f}")

    print("\n--- 6. PER-CLASS METRICS ---")
    cls_report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    def get_cm_val(true_name, pred_name):
        t = CLASS_NAMES.index(true_name)
        p = CLASS_NAMES.index(pred_name)
        return int(cm[t, p])

    print("\n--- 9. KEY MISCLASSIFICATIONS ---")
    print(f"nv -> mel count: {get_cm_val('nv', 'mel')}")
    print(f"mel -> nv count: {get_cm_val('mel', 'nv')}")
    print(f"nv -> bkl count: {get_cm_val('nv', 'bkl')}")
    print(f"bkl -> nv count: {get_cm_val('bkl', 'nv')}")
    print(f"bkl -> mel count: {get_cm_val('bkl', 'mel')}")

    mean_conf = float(np.mean(all_max_confs))
    median_conf = float(np.median(all_max_confs))
    low_conf_count = int(sum(c < 0.5 for c in all_max_confs))
    low_conf_pct = (low_conf_count / len(all_max_confs)) * 100

    print(f"\n10. Mean max confidence: {mean_conf:.4f}")
    print(f"11. Median max confidence: {median_conf:.4f}")
    print(f"12. Low-confidence predictions (<0.5): {low_conf_count} / {len(all_max_confs)} ({low_conf_pct:.2f}%)")

    # SAVE JSON
    eval_data = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": cls_report,
        "key_misclassifications": {
            "nv_to_mel": get_cm_val('nv', 'mel'),
            "mel_to_nv": get_cm_val('mel', 'nv'),
            "nv_to_bkl": get_cm_val('nv', 'bkl'),
            "bkl_to_nv": get_cm_val('bkl', 'nv'),
            "bkl_to_mel": get_cm_val('bkl', 'mel')
        },
        "confidence": {
            "mean_max_conf": mean_conf,
            "median_max_conf": median_conf,
            "low_conf_count": low_conf_count,
            "low_conf_pct": low_conf_pct
        }
    }
    json_path = PROJECT_ROOT / "models" / "evaluation_resnet18_softweights_finetuned.json"
    with open(json_path, 'w') as f:
        json.dump(eval_data, f, indent=4)
    print(f"\nSaved evaluation JSON to {json_path}")

    # Save artifacts
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
        
    csv_path = PROJECT_ROOT / "models" / "test_predictions_resnet18_softweights_finetuned.csv"
    preds_df.to_csv(csv_path, index=False)
    print(f"Saved test predictions to {csv_path}")

    # Confusion matrices
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Phase 2 Test Confusion Matrix (Raw)')
    raw_cm_path = PROJECT_ROOT / "models" / "confusion_matrix_resnet18_softweights_finetuned.png"
    plt.savefig(raw_cm_path)
    plt.close()
    print(f"Saved raw confusion matrix to {raw_cm_path}")

    cm_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Phase 2 Test Confusion Matrix (Normalized)')
    norm_cm_path = PROJECT_ROOT / "models" / "confusion_matrix_resnet18_softweights_finetuned_normalized.png"
    plt.savefig(norm_cm_path)
    plt.close()
    print(f"Saved normalized confusion matrix to {norm_cm_path}")

if __name__ == "__main__":
    main()
