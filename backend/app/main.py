"""
app/main.py — FastAPI application entry point.

Creates the app, configures CORS, mounts routers, and handles
model loading/unloading via the FastAPI lifespan context.

Start the server:
    .venv\\Scripts\\uvicorn app.main:app --reload --port 8000

Production start (in container):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging
from app.routes import health, predict
from app.services.model_service import model_service

# ---------------------------------------------------------------------------
# Logging — configure before any module-level loggers fire
# ---------------------------------------------------------------------------
setup_logging(debug=settings.debug)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — model load/unload tied to app lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    On startup:  Load the PyTorch model into memory.
    On shutdown: Release the model.

    Using lifespan instead of @app.on_event("startup") because the latter
    is deprecated in FastAPI ≥ 0.93.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)
    logger.info("Model checkpoint: %s", settings.model_path)
    logger.info("Allowed origins: %s", settings.allowed_origins_list)
    logger.info("=" * 60)

    try:
        model_service.load(settings.model_path)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        # Model checkpoint not found — app still starts but /predict will return 503.
        # This allows the container to start and /health to return a useful response
        # even before a trained model is placed in the models/ directory.
        logger.warning(
            "Model checkpoint not found at: %s\n"
            "The /predict endpoint will return 503 until a model is available.\n"
            "Run a training script to produce a checkpoint.",
            settings.model_path,
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        logger.warning("API will start without a model. /predict returns 503.")

    yield  # ← application runs here

    # Shutdown
    logger.info("Shutting down — releasing model...")
    model_service.unload()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "## Skin Lesion Classification API\n\n"
        "An educational AI tool for classifying skin lesion images into "
        "7 categories from the HAM10000 dataset using a trained PyTorch model.\n\n"
        "### Classes\n"
        "| Code | Full Name |\n"
        "|------|----------|\n"
        "| `akiec` | Actinic keratoses and intraepithelial carcinoma |\n"
        "| `bcc` | Basal cell carcinoma |\n"
        "| `bkl` | Benign keratinocytic lesions |\n"
        "| `df` | Dermatofibroma |\n"
        "| `mel` | Melanoma |\n"
        "| `nv` | Melanocytic nevi |\n"
        "| `vasc` | Vascular lesions |\n\n"
        "### ⚠️ Medical Disclaimer\n"
        "This API is for **educational and research purposes only**. "
        "It is **NOT** a medical diagnostic tool and results must **not** be "
        "used to make clinical decisions. Always consult a qualified dermatologist."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — configurable for separate React frontend
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(predict.router)


# ---------------------------------------------------------------------------
# Global exception handler — catch any unhandled exception, log it,
# return a clean JSON 500 (no stack traces to the client)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )
