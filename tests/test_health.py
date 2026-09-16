"""
tests/test_health.py — Tests for GET /health.
"""

from fastapi.testclient import TestClient


def test_health_returns_200(client_with_model: TestClient) -> None:
    """Health endpoint must return HTTP 200."""
    response = client_with_model.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client_with_model: TestClient) -> None:
    """Health response must contain required fields."""
    response = client_with_model.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "app_version" in data


def test_health_status_healthy(client_with_model: TestClient) -> None:
    """status field must be 'healthy'."""
    response = client_with_model.get("/health")
    assert response.json()["status"] == "healthy"


def test_health_model_loaded_true(client_with_model: TestClient) -> None:
    """model_loaded must be True when the mock model is active."""
    response = client_with_model.get("/health")
    assert response.json()["model_loaded"] is True


def test_health_model_type_present(client_with_model: TestClient) -> None:
    """model_type must be reported when model is loaded."""
    response = client_with_model.get("/health")
    assert response.json()["model_type"] == "resnet18"
