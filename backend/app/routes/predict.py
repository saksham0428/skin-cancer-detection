"""
app/routes/predict.py — POST /predict endpoint.

Accepts an uploaded skin lesion image, validates it, runs inference,
and returns a structured prediction response.

Error handling:
    422  No file uploaded (FastAPI automatic validation)
    415  Unsupported file content type
    413  File too large
    400  Image cannot be opened / decoded (corrupt file)
    503  Model not loaded
    500  Unexpected inference failure

All errors return JSON with "error" and "detail" keys.
No internal stack traces are ever exposed to the client.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.prediction import ErrorResponse, PredictionResponse
from app.services.model_service import model_service
from src.config import ALLOWED_IMAGE_CONTENT_TYPES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Corrupt or unreadable image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
        500: {"model": ErrorResponse, "description": "Internal inference error"},
    },
    summary="Classify a skin lesion image",
    description=(
        "Upload a skin lesion image (JPEG, PNG, WebP, or BMP). "
        "The model returns the predicted lesion class, confidence score, "
        "and probability distribution over all 7 HAM10000 classes.\n\n"
        "**Important:** This is an educational research tool. "
        "Results are **not** a medical diagnosis. "
        "Always consult a qualified dermatologist."
    ),
    tags=["Prediction"],
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Skin lesion image file (JPEG, PNG, WebP, or BMP). Max 10 MB.",
    ),
) -> PredictionResponse:
    """
    Run skin lesion classification on an uploaded image.

    The image is preprocessed identically to the validation pipeline used
    during model training:
      1. Resize to 224×224
      2. Convert to tensor
      3. Normalise with ImageNet mean/std

    Returns probability scores over 7 HAM10000 lesion categories.
    """
    # --- 1. Check model readiness ---
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_not_ready",
                "detail": "The classification model is not loaded. The service is starting up.",
            },
        )

    # --- 2. Validate content type ---
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": "invalid_file_type",
                "detail": (
                    f"Uploaded file type '{content_type}' is not supported. "
                    "Please upload a JPEG, PNG, WebP, or BMP image."
                ),
            },
        )

    # --- 3. Read file content ---
    image_bytes = await file.read()

    # --- 4. Validate file size ---
    max_bytes = settings.max_upload_size_bytes
    if len(image_bytes) > max_bytes:
        max_mb = settings.max_upload_size_mb
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "detail": (
                    f"Uploaded file is {len(image_bytes) / 1024 / 1024:.1f} MB. "
                    f"Maximum allowed size is {max_mb:.0f} MB."
                ),
            },
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "empty_file",
                "detail": "Uploaded file is empty.",
            },
        )

    # --- 5. Run inference ---
    logger.info(
        "Inference request: file='%s' type='%s' size=%d bytes",
        file.filename,
        content_type,
        len(image_bytes),
    )

    try:
        result = model_service.predict(image_bytes)
    except ValueError as exc:
        # Image could not be decoded — corrupt or unsupported format
        logger.warning("Image decode failed for '%s': %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "corrupt_image",
                "detail": (
                    "The uploaded image could not be opened or decoded. "
                    "Please ensure the file is a valid, uncorrupted image."
                ),
            },
        )
    except RuntimeError as exc:
        logger.error("Model service error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_not_ready",
                "detail": str(exc),
            },
        )
    except Exception as exc:
        # Catch-all — never expose stack traces
        logger.exception("Unexpected inference error for '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "inference_failed",
                "detail": "An unexpected error occurred during inference. Please try again.",
            },
        )

    logger.info(
        "Prediction: class='%s' confidence=%.4f file='%s'",
        result.predicted_class,
        result.confidence,
        file.filename,
    )

    return PredictionResponse(
        predicted_class=result.predicted_class,
        predicted_class_full_name=result.predicted_class_full_name,
        class_index=result.class_index,
        confidence=result.confidence,
        probabilities=result.probabilities,
        disclaimer=result.disclaimer,
    )
