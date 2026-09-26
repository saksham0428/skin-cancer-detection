"""
app/schemas/health.py — Response schema for the health endpoint.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    model_loaded: bool
    model_type: str | None = None
    app_version: str

    model_config = {"json_schema_extra": {
        "example": {
            "status": "healthy",
            "model_loaded": True,
            "model_type": "resnet18",
            "app_version": "1.0.0",
        }
    }}
