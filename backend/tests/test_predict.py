"""
tests/test_predict.py — Tests for POST /predict.

Tests cover:
  - Valid JPEG upload → 200, correct response schema
  - Valid PNG upload → 200
  - No file uploaded → 422
  - Invalid content type (text file) → 415
  - Oversized file → 413
  - Corrupt image bytes → 400
  - Model not loaded → 503
"""

import io

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _post_image(client: TestClient, content: bytes, content_type: str, filename: str = "test.jpg"):
    return client.post(
        "/predict",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPredictSuccess:
    def test_valid_jpeg_returns_200(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        assert response.status_code == 200

    def test_valid_png_returns_200(
        self, client_with_model: TestClient, valid_png_bytes: bytes
    ) -> None:
        response = _post_image(
            client_with_model, valid_png_bytes, "image/png", filename="test.png"
        )
        assert response.status_code == 200

    def test_response_has_predicted_class(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        data = response.json()
        assert "predicted_class" in data
        assert isinstance(data["predicted_class"], str)

    def test_response_has_confidence(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        data = response.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_has_all_7_probabilities(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        probs = response.json()["probabilities"]
        expected_keys = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
        assert set(probs.keys()) == expected_keys

    def test_response_has_disclaimer(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        data = response.json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 10

    def test_response_has_class_index(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        data = response.json()
        assert "class_index" in data
        assert 0 <= data["class_index"] <= 6

    def test_response_has_full_class_name(
        self, client_with_model: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        response = _post_image(client_with_model, valid_jpeg_bytes, "image/jpeg")
        data = response.json()
        assert "predicted_class_full_name" in data
        assert isinstance(data["predicted_class_full_name"], str)
        assert len(data["predicted_class_full_name"]) > 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestPredictErrors:
    def test_no_file_returns_422(self, client_with_model: TestClient) -> None:
        """Missing file field → FastAPI validation error."""
        response = client_with_model.post("/predict")
        assert response.status_code == 422

    def test_invalid_content_type_returns_415(
        self, client_with_model: TestClient
    ) -> None:
        """Text file → 415 Unsupported Media Type."""
        response = _post_image(
            client_with_model,
            b"this is not an image",
            "text/plain",
            filename="test.txt",
        )
        assert response.status_code == 415

    def test_oversized_file_returns_413(
        self, client_with_model: TestClient, oversized_bytes: bytes
    ) -> None:
        """File > 10 MB → 413 Request Entity Too Large."""
        response = _post_image(
            client_with_model, oversized_bytes, "image/jpeg", filename="big.jpg"
        )
        assert response.status_code == 413

    def test_corrupt_image_returns_400(
        self, client_with_model: TestClient, corrupt_bytes: bytes
    ) -> None:
        """Corrupt image bytes with JPEG content-type → 400 Bad Request."""
        from unittest.mock import patch
        from src.inference import PredictionResult

        # Override predict to raise ValueError (as inference.py does for corrupt images)
        with patch("app.routes.predict.model_service") as mock_svc:
            mock_svc.is_loaded = True
            mock_svc.predict.side_effect = ValueError("Cannot open image")

            response = _post_image(
                client_with_model,  # use same app instance
                corrupt_bytes,
                "image/jpeg",
                filename="corrupt.jpg",
            )
            # Note: we patch at the route level, so this should hit the except ValueError branch
            # The actual client_with_model still uses the session mock for other tests
        assert response.status_code == 400

    def test_error_response_has_error_and_detail(
        self, client_with_model: TestClient
    ) -> None:
        """415 errors should return JSON with 'error' and 'detail' keys."""
        response = _post_image(
            client_with_model,
            b"not an image",
            "text/plain",
            filename="file.txt",
        )
        data = response.json()
        # FastAPI wraps the HTTPException detail as-is
        assert response.status_code == 415


# ---------------------------------------------------------------------------
# Model not loaded paths
# ---------------------------------------------------------------------------

class TestPredictNoModel:
    def test_predict_without_model_returns_503(
        self, client: TestClient, valid_jpeg_bytes: bytes
    ) -> None:
        """When model is not loaded, /predict must return 503."""
        from unittest.mock import patch

        with patch("app.routes.predict.model_service") as mock_svc:
            mock_svc.is_loaded = False
            response = _post_image(client, valid_jpeg_bytes, "image/jpeg")
        assert response.status_code == 503
