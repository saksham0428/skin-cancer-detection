"""
tests/conftest.py — Shared pytest fixtures.

Fixtures available to all test modules:
  - client: FastAPI TestClient (model NOT loaded — tests model-absent paths)
  - client_with_model: TestClient with a mock model loaded
  - valid_jpeg_bytes: Minimal valid JPEG bytes
  - valid_png_bytes: Minimal valid PNG bytes
  - oversized_bytes: Bytes that exceed the upload limit
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Make sure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_image_bytes(fmt: str = "JPEG", size: tuple = (100, 100)) -> bytes:
    """Create a minimal valid image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(128, 64, 32))
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def valid_jpeg_bytes() -> bytes:
    return _make_image_bytes("JPEG")


@pytest.fixture(scope="session")
def valid_png_bytes() -> bytes:
    return _make_image_bytes("PNG")


@pytest.fixture(scope="session")
def oversized_bytes() -> bytes:
    """Bytes larger than the 10MB upload limit."""
    return b"x" * (11 * 1024 * 1024)  # 11 MB


@pytest.fixture(scope="session")
def corrupt_bytes() -> bytes:
    """Bytes that look like a JPEG header but are corrupt."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100  # truncated JPEG


# ---------------------------------------------------------------------------
# Client WITHOUT a loaded model (tests service-unavailable paths)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    TestClient with the FastAPI app.

    The model is NOT loaded — useful for testing /health (model_loaded=False)
    and 503 paths.
    """
    from app.main import app
    # Patch lifespan to skip actual model loading
    with patch("app.main.model_service") as mock_svc:
        mock_svc.is_loaded = False
        mock_svc.model_type = None
        mock_svc.unload = MagicMock()
        # Temporarily override the lifespan so it doesn't try to load
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def null_lifespan(app):
            yield

        app.router.lifespan_context = null_lifespan
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Client WITH a mock model loaded (tests the happy path)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client_with_model(valid_jpeg_bytes) -> TestClient:
    """
    TestClient where model_service.predict returns a fixed mock result.
    """
    from app.main import app
    from src.inference import PredictionResult
    from src.config import MEDICAL_DISCLAIMER

    mock_result = PredictionResult(
        predicted_class="nv",
        predicted_class_full_name="Melanocytic nevi",
        class_index=5,
        confidence=0.8234,
        probabilities={
            "akiec": 0.0200,
            "bcc":   0.0150,
            "bkl":   0.0516,
            "df":    0.0100,
            "mel":   0.0500,
            "nv":    0.8234,
            "vasc":  0.0300,
        },
        disclaimer=MEDICAL_DISCLAIMER,
    )

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def null_lifespan(app):
        yield

    app.router.lifespan_context = null_lifespan

    with patch("app.routes.predict.model_service") as mock_svc:
        mock_svc.is_loaded = True
        mock_svc.model_type = "resnet18"
        mock_svc.predict = MagicMock(return_value=mock_result)

        with patch("app.routes.health.model_service") as mock_health_svc:
            mock_health_svc.is_loaded = True
            mock_health_svc.model_type = "resnet18"

            yield TestClient(app)
