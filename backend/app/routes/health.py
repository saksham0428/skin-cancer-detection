"""
app/routes/health.py — GET /health endpoint.

Returns current service status and whether the model is loaded.
Used by load balancers, Docker HEALTHCHECK, and monitoring.
"""

import logging

from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import HealthResponse
from app.services.model_service import model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the current health status of the API and whether the "
        "classification model is loaded and ready to accept predictions."
    ),
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """
    Check if the API is running and the model is ready.

    A 200 response means the service is reachable.
    Check `model_loaded` to confirm the model is ready for predictions.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        model_type=model_service.model_type,
        app_version=settings.app_version,
    )
