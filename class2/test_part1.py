"""
Part 1 — Interface Layer Tests
Tests for api_simple.py: single-model prediction, input validation, error handling.
Run: pytest class2/test_part1.py -v
"""

import io
import time
from PIL import Image
from fastapi.testclient import TestClient

from api_simple import app

client = TestClient(app)


# ======================== Test Image Helpers ========================

def create_test_image(width=100, height=100, color=(255, 0, 0), fmt="PNG"):
    """Create a valid test image in memory"""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def create_corrupt_file():
    """Create a file with invalid image data"""
    buf = io.BytesIO(b"this is not an image at all")
    buf.seek(0)
    return buf


# ======================== Health & Info Tests ========================

class TestHealthAndInfo:
    """Test health check and model info endpoints"""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_includes_fields(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model_version" in data
        assert "device" in data
        assert "timestamp" in data

    def test_info_returns_200(self):
        response = client.get("/info")
        assert response.status_code == 200

    def test_info_includes_classes(self):
        data = client.get("/info").json()
        assert data["num_classes"] == 7
        assert len(data["classes"]) == 7
        assert "animal" in data["classes"]
        assert "vehicle" in data["classes"]
        assert data["input_shape"] == "128x128x3"


# ======================== Valid Prediction Tests ========================

class TestValidPredictions:
    """Test successful predictions with valid images"""

    def test_valid_png_returns_200(self):
        img = create_test_image(100, 100, (255, 0, 0), "PNG")
        response = client.post("/predict", files={"file": ("test.png", img, "image/png")})
        assert response.status_code == 200

    def test_response_has_required_fields(self):
        img = create_test_image()
        response = client.post("/predict", files={"file": ("test.png", img, "image/png")})
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_confidence_in_valid_range(self):
        img = create_test_image()
        data = client.post("/predict", files={"file": ("test.png", img, "image/png")}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_class_probabilities_sum_to_one(self):
        img = create_test_image()
        data = client.post("/predict", files={"file": ("test.png", img, "image/png")}).json()
        probs = data["class_probabilities"]
        assert len(probs) == 7
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_prediction_is_valid_class(self):
        img = create_test_image()
        data = client.post("/predict", files={"file": ("test.png", img, "image/png")}).json()
        valid_classes = {'animal', 'name_board', 'pedestrian', 'pothole', 'road_sign', 'speed_breaker', 'vehicle'}
        assert data["prediction"] in valid_classes

    def test_large_image_handled(self):
        """Large images should be resized and predicted successfully"""
        img = create_test_image(1024, 1024)
        response = client.post("/predict", files={"file": ("large.png", img, "image/png")})
        assert response.status_code == 200

    def test_valid_jpeg(self):
        img = create_test_image(100, 100, (0, 255, 0), "JPEG")
        response = client.post("/predict", files={"file": ("test.jpg", img, "image/jpeg")})
        assert response.status_code == 200


# ======================== Error Handling Tests ========================

class TestErrorHandling:
    """Test input validation and error responses"""

    def test_invalid_file_type_returns_400(self):
        """Non-image file types should be rejected"""
        buf = io.BytesIO(b"hello world")
        response = client.post("/predict", files={"file": ("test.txt", buf, "text/plain")})
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_corrupt_image_returns_400(self):
        """Corrupt image data should be rejected"""
        buf = create_corrupt_file()
        response = client.post("/predict", files={"file": ("corrupt.png", buf, "image/png")})
        assert response.status_code == 400

    def test_empty_file_returns_400(self):
        """Empty file should be rejected"""
        buf = io.BytesIO(b"")
        response = client.post("/predict", files={"file": ("empty.png", buf, "image/png")})
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_tiny_image_returns_400(self):
        """Images smaller than 32x32 should be rejected"""
        img = create_test_image(16, 16)
        response = client.post("/predict", files={"file": ("tiny.png", img, "image/png")})
        assert response.status_code == 400
        assert "too small" in response.json()["detail"]

    def test_missing_file_returns_422(self):
        """Missing file parameter should return 422"""
        response = client.post("/predict")
        assert response.status_code == 422


# ======================== Performance Tests ========================

class TestPerformance:
    """Test basic performance requirements"""

    def test_prediction_latency_under_5s(self):
        """Single prediction should complete within 5 seconds"""
        img = create_test_image()
        start = time.time()
        response = client.post("/predict", files={"file": ("test.png", img, "image/png")})
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 5.0

    def test_sequential_requests(self):
        """API should handle 10 sequential requests without errors"""
        for i in range(10):
            img = create_test_image(100, 100, (i * 25, 0, 0))
            response = client.post("/predict", files={"file": (f"test_{i}.png", img, "image/png")})
            assert response.status_code == 200
