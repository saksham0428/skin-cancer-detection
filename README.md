# Skin Lesion Classification — AI Backend

An educational AI backend for classifying dermatoscopic skin lesion images using the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Built with PyTorch and FastAPI, designed for deployment as a standalone REST API service that a React frontend can consume.

> ⚠️ **Medical Disclaimer:** This project is for **educational and research purposes only**. It is **not** a medical diagnostic tool. Results must **not** be used to make clinical decisions. Always consult a qualified dermatologist for any skin concerns.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dataset](#dataset)
3. [Class Labels](#class-labels)
4. [Preprocessing](#preprocessing)
5. [Leakage-Safe Data Split](#leakage-safe-data-split)
6. [Model Architecture](#model-architecture)
7. [Class Imbalance Handling](#class-imbalance-handling)
8. [Training](#training)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Inference Pipeline](#inference-pipeline)
11. [API Endpoints](#api-endpoints)
12. [Local Setup](#local-setup)
13. [Environment Variables](#environment-variables)
14. [Docker Usage](#docker-usage)
15. [Running Tests](#running-tests)
16. [Project Structure](#project-structure)
17. [Limitations](#limitations)

---

## Architecture Overview

```
React Frontend
     │
     │  HTTP POST /predict (multipart/form-data image)
     ▼
FastAPI Backend (app/)
     │
     ├── Request validation (content type, size, empty)
     ├── Image decode (PIL)
     ├── Preprocessing (resize 224×224, ImageNet normalisation)
     ├── Model inference (PyTorch)
     ├── Softmax probabilities
     └── Structured JSON response
```

---

## Dataset

**HAM10000** (Human Against Machine with 10,000 training images)

- 10,015 dermatoscopic images across 7 skin lesion classes
- Images are JPG format, variable size (resized to 224×224 during preprocessing)
- Metadata includes: `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, `localization`
- Source: ISIC Archive / Kaggle

---

## Class Labels

| Index | Code | Full Name |
|-------|------|-----------|
| 0 | `akiec` | Actinic keratoses and intraepithelial carcinoma |
| 1 | `bcc` | Basal cell carcinoma |
| 2 | `bkl` | Benign keratinocytic lesions |
| 3 | `df` | Dermatofibroma |
| 4 | `mel` | Melanoma |
| 5 | `nv` | Melanocytic nevi |
| 6 | `vasc` | Vascular lesions |

**Class distribution (train split):**

| Class | Count | % of train |
|-------|-------|-----------|
| nv | 5363 | 66.9% |
| mel | 890 | 11.1% |
| bkl | 879 | 11.0% |
| bcc | 411 | 5.1% |
| akiec | 262 | 3.3% |
| vasc | 114 | 1.4% |
| df | 92 | 1.1% |

The dataset is **severely imbalanced** (nv:df ratio ≈ 58:1).

---

## Preprocessing

Both training and inference use the same base pipeline (consistency is critical):

```
Input image
→ PIL.Image.open().convert("RGB")     # Always RGB, handles RGBA/grayscale
→ transforms.Resize((224, 224))        # Bilinear resize
→ transforms.ToTensor()                # [0,255] → [0.0,1.0]
→ transforms.Normalize(                # ImageNet mean/std
      mean=[0.485, 0.456, 0.406],
      std =[0.229, 0.224, 0.225]
  )
```

**Training additionally applies:**
- `RandomHorizontalFlip(p=0.5)`
- `RandomVerticalFlip(p=0.5)`
- `RandomRotation(15°)`

Inference uses exactly the same pipeline as validation — no preprocessing mismatch is possible because both import from `src/transforms.py`.

---

## Leakage-Safe Data Split

The HAM10000 dataset contains **repeated images of the same lesion** (up to 6 images per lesion). A naive random split would create **data leakage** — the same lesion appearing in both train and test.

**Solution used:** `StratifiedGroupKFold` with `lesion_id` as the group variable.

```python
StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
```

This ensures:
- No `lesion_id` appears in more than one split
- Class proportions are preserved across splits
- Split is deterministic (fixed `random_state=42`)

**Verified:**
```
Train ∩ Validation  = 0 unique lesion_ids
Train ∩ Test        = 0 unique lesion_ids
Validation ∩ Test   = 0 unique lesion_ids
```

**Final split sizes:**

| Split | Images |
|-------|--------|
| Train | 8,011 |
| Validation | 1,002 |
| Test | 1,002 |

---

## Model Architecture

### Baseline — SimpleCNN

Trained from scratch on HAM10000.

```
Input: [B, 3, 224, 224]
Conv2d(3→32, 3×3) + ReLU + MaxPool(2)  →  [B, 32, 112, 112]
Conv2d(32→64, 3×3) + ReLU + MaxPool(2) →  [B, 64,  56,  56]
Conv2d(64→128, 3×3) + ReLU + MaxPool(2)→  [B, 128, 28,  28]
Flatten                                 →  [B, 100352]
Linear(100352→256) + ReLU + Dropout(0.5)
Linear(256→7)                           → logits [B, 7]
```

Total parameters: **~25.8M**

### Transfer Learning — ResNet18 (Final Fine-tuned Model)

ImageNet pretrained ResNet18, fully fine-tuned on HAM10000.

**Strategy:**
- **Phase 1 (Frozen):** Froze all layers except `layer4` and the FC head. Handled class imbalance using heavily softened weights.
- **Phase 2 (Fine-tuning):** Unfroze the entire backbone. Used differential learning rates (1e-5 for the backbone, 1e-4 for the classification head).
- Checkpoint: `models/skin_lesion_resnet18_softweights_finetuned.pth`

**Why ResNet18 over ResNet50 or EfficientNet?**
- CPU-only / DirectML limitations — ResNet50 would be 3–5× slower per epoch
- EfficientNet requires the `timm` library (new dependency)
- ResNet18 is well-studied for medical imaging and achieves competitive results on HAM10000

---

## Class Imbalance Handling

**Strategy: Class-Weighted CrossEntropyLoss (Softened)**

Weights are computed using the sklearn "balanced" formula, but we introduced a "softening" factor by taking the square root to prevent the model from over-penalizing majority classes and collapsing.

| Class | Count | Softened Weight |
|-------|-------|--------|
| nv | 5363 | 0.25 |
| mel | 890 | 0.60 |
| bkl | 879 | 0.61 |
| bcc | 411 | 0.89 |
| akiec | 262 | 1.11 |
| vasc | 114 | 1.68 |
| df | 92 | 1.87 |

**Primary validation metric: Macro F1** (not accuracy)

Accuracy is a misleading metric for imbalanced data. A model predicting only `nv` would achieve ~67% train accuracy but essentially zero F1 for the other 6 classes. Early stopping and model selection use **macro F1** instead.

---

## Training

### Configuration

| Hyperparameter | SimpleCNN | ResNet18 (Phase 2 Fine-tuned) |
|---------------|-----------|---------|
| Epochs | 15 | 15 (Early Stopped at 6) |
| Learning Rate | 1e-3 | Head: 1e-4, Backbone: 1e-5 |
| Weight Decay | 1e-4 | 1e-4 |
| Batch Size | 32 | 32 |
| Optimizer | Adam | Adam |
| Early Stopping | patience=5 on val macro F1 | patience=5 |
| Class Weighting | ✓ | ✓ (Softened) |

### Running Training

```bash
# Activate virtual environment
.venv\Scripts\activate

# Train Phase 1
python scripts/train_resnet18_softweights.py

# Train Phase 2 (Fine-tuning)
python scripts/finetune_resnet18_softweights.py
```

Training checkpoints are saved to `models/` automatically.

---

## Final Model Evaluation Metrics

After locking the final model (`models/skin_lesion_resnet18_softweights_finetuned.pth`), we evaluated it exactly once on the sealed 1002-image Test Set:

- **Overall Test Accuracy:** 82.63%
- **Macro Precision:** 72.05%
- **Macro Recall:** 67.03%
- **Macro F1:** **69.05%**
- **Weighted F1:** 82.42%

*Key error reduction:* The dangerous `nv → mel` misclassifications were reduced to just 31 instances out of 1002 test images.

> **Important:** Because this dataset is severely imbalanced, **macro F1 is the primary metric**. Weighted metrics are biased toward the majority class (`nv`) and can mask poor performance on rare but clinically important classes.

---

## Inference Pipeline

The inference module (`src/inference.py`) is fully independent of any notebook:

```python
from src.inference import load_model, predict_image

model = load_model("models/skin_lesion_resnet18.pth")
result = predict_image(image_bytes, model)

print(result.predicted_class)        # e.g. "mel"
print(result.confidence)             # e.g. 0.7821
print(result.probabilities)          # all 7 classes
print(result.disclaimer)             # medical disclaimer
```

---

## API Endpoints

### `GET /health`

Returns the service status and whether the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "resnet18",
  "app_version": "1.0.0"
}
```

---

### `POST /predict`

Upload a skin lesion image for classification.

**Request:** `multipart/form-data` with a `file` field (JPEG, PNG, WebP, or BMP, max 10 MB).

**Response (200 OK):**
```json
{
  "predicted_class": "mel",
  "predicted_class_full_name": "Melanoma",
  "class_index": 4,
  "confidence": 0.7821,
  "probabilities": {
    "akiec": 0.0312,
    "bcc":   0.0241,
    "bkl":   0.0581,
    "df":    0.0093,
    "mel":   0.7821,
    "nv":    0.0841,
    "vasc":  0.0111
  },
  "disclaimer": "This result is generated by a research AI model for educational purposes only. ..."
}
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Corrupt or unreadable image |
| 413 | File exceeds 10 MB |
| 415 | Unsupported file type (not JPEG/PNG/WebP/BMP) |
| 422 | Missing `file` field |
| 503 | Model not loaded |
| 500 | Unexpected inference failure |

---

## Local Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd skin-cancer-detection

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements-backend.txt

# 4. Copy environment template
cp .env.example .env
# Edit .env if needed

# 5. Place a trained model checkpoint in models/
# (or run a training script first — see Training section)

# 6. Start the API
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` or `production` |
| `DEBUG` | `False` | Enable debug logging |
| `MODEL_TYPE` | `resnet18` | Model architecture (`resnet18` or `simple_cnn`) |
| `MODEL_CHECKPOINT` | `models/skin_lesion_resnet18.pth` | Path to checkpoint file |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated CORS origins |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum upload file size in MB |

See `.env.example` for the complete template.

---

## Docker Usage

### Build

```bash
docker build -t skin-cancer-backend .
```

> **Note:** The `.dockerignore` excludes `models/` by default. To bake the model into the image, comment out the `models/` line in `.dockerignore` before building, or use a volume mount (recommended for production).

### Run with volume mount (recommended)

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_CHECKPOINT=models/skin_lesion_resnet18.pth \
  -e ALLOWED_ORIGINS=http://localhost:3000 \
  skin-cancer-backend
```

### Run with model baked into image

```bash
# 1. Edit .dockerignore to allow models/
# 2. Rebuild
docker build -t skin-cancer-backend .
docker run -p 8000:8000 skin-cancer-backend
```

### Test the container

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

---

## Running Tests

```bash
# From project root with venv active
.venv\Scripts\python.exe -m pytest tests/ -v
```

Test coverage:
- `GET /health` — status, schema, model_loaded
- `POST /predict` — valid JPEG, valid PNG, response schema, all 7 probabilities, disclaimer
- Error handling — 422 (no file), 415 (wrong type), 413 (too large), 400 (corrupt), 503 (no model)

---

## Project Structure

```
skin-cancer-detection/
├── src/                     # Core ML modules (training + inference)
│   ├── config.py            # All constants: paths, class names, hyperparams
│   ├── dataset.py           # SkinLesionDataset + compute_class_weights
│   ├── transforms.py        # Train/val/inference pipelines
│   ├── model.py             # SimpleCNN + build_resnet18()
│   ├── train.py             # Training loop, early stopping, checkpointing
│   ├── evaluate.py          # Metrics + confusion matrix
│   └── inference.py         # Production inference pipeline
├── app/                     # FastAPI backend
│   ├── main.py              # App creation, CORS, lifespan
│   ├── core/
│   │   ├── config.py        # Pydantic Settings (env vars)
│   │   └── logging.py       # Logging setup
│   ├── routes/
│   │   ├── health.py        # GET /health
│   │   └── predict.py       # POST /predict
│   ├── services/
│   │   └── model_service.py # Singleton model loader
│   └── schemas/
│       ├── health.py        # HealthResponse
│       └── prediction.py    # PredictionResponse, ErrorResponse
├── scripts/                 # Training and evaluation CLI scripts
│   ├── train_simple_cnn.py
│   ├── train_resnet18.py
│   └── evaluate_model.py
├── models/                  # Saved checkpoints (gitignored .pth files)
├── notebooks/               # Exploratory notebooks (EDA, dataset, first CNN)
├── data/                    # Dataset (gitignored)
├── tests/                   # Pytest tests
├── .env.example             # Environment template
├── requirements-backend.txt # Minimal production dependencies
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## Limitations

1. **CPU-only training and inference.** This machine has no NVIDIA GPU. Training is slow (~30–60 min for SimpleCNN, ~2–4 hours for ResNet18 per training run). Inference on CPU adds latency (~100–500ms per image).

2. **HAM10000 dataset limitations.** The dataset is imbalanced (66% `nv`) and was collected in a clinical setting with dermoscopes. Model performance on consumer smartphone photos may differ.

3. **Not a medical device.** This model has not been clinically validated. It must not be used for diagnosis.

4. **7 classes only.** The model cannot detect lesion types outside the HAM10000 taxonomy.

5. **Macro F1 is the primary metric, not accuracy.** Due to class imbalance, overall accuracy is misleading. A model predicting only `nv` would achieve ~67% accuracy but essentially zero F1 on rare classes.

6. **Single-model serving.** The container runs `--workers 1` because PyTorch models are not safe to share across forked processes. For higher throughput, run multiple containers behind a load balancer.
