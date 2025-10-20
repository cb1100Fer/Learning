"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_image_file():
    """Create sample image file."""
    img = Image.new('RGB', (224, 224), color='green')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/model/info")
    # May return 503 if model not loaded in test
    assert response.status_code in [200, 503]


def test_predict_endpoint_invalid_file(client):
    """Test prediction with invalid file."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code in [400, 503]  # 400 for bad file, 503 if no model


# Note: Full prediction tests would require a loaded model
# which is not available in unit tests without proper setup
