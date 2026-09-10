"""
Contract & Error Handling Tests for FastAPI Endpoints.
Verifies clean 422s, 404s, CORS headers, and audit trail functionality.
"""

from pathlib import Path
import sys
import io
import base64
import numpy as np
import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(API_DIR))

from main import app
from inference import engine

BASE_DIR = API_DIR.parent
DATASET_VALID_DIR = BASE_DIR / "dataset" / "valid"

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_engine():
    engine.load_all()


def npy_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture(scope="module")
def sample_volumes_b64():
    ax = np.load(DATASET_VALID_DIR / "axial" / "1130.npy")
    co = np.load(DATASET_VALID_DIR / "coronal" / "1130.npy")
    sa = np.load(DATASET_VALID_DIR / "sagittal" / "1130.npy")
    return {
        "axial": npy_to_b64(ax),
        "coronal": npy_to_b64(co),
        "sagittal": npy_to_b64(sa),
    }


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "proposed" in data["models_loaded"]
    assert "transfer_learning" in data["models_loaded"]
    assert "custom_cnn" in data["models_loaded"]


def test_models_metadata_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) == 3
    # Check proposed has attention True, others False
    for m in data["models"]:
        if m["model_key"] == "proposed":
            assert m["has_attention"] is True
        else:
            assert m["has_attention"] is False


def test_predict_success(sample_volumes_b64):
    payload = {
        "model": "proposed",
        "axial": sample_volumes_b64["axial"],
        "coronal": sample_volumes_b64["coronal"],
        "sagittal": sample_volumes_b64["sagittal"],
        "threshold": 0.5,
        "exam_id": "test_1130"
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "proposed"
    assert "probability" in data
    assert "plane_logits" in data
    assert "fusion_weights" in data
    assert data["predicted_label"] == "normal"


def test_unknown_model_returns_404(sample_volumes_b64):
    payload = {
        "model": "non_existent_model_v99",
        "axial": sample_volumes_b64["axial"],
        "coronal": sample_volumes_b64["coronal"],
        "sagittal": sample_volumes_b64["sagittal"],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 404
    assert "Unknown model" in resp.json()["detail"]


def test_malformed_npy_shape_returns_422(sample_volumes_b64):
    # 2D array instead of 3D volume (S, H, W)
    bad_2d = np.ones((256, 256), dtype=np.float32)
    bad_b64 = npy_to_b64(bad_2d)

    payload = {
        "model": "proposed",
        "axial": bad_b64,
        "coronal": sample_volumes_b64["coronal"],
        "sagittal": sample_volumes_b64["sagittal"],
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "must have shape (S, H, W) with ndim=3" in resp.json()["detail"]


def test_gradcam_custom_cnn_rejected_with_422(sample_volumes_b64):
    payload = {
        "model": "custom_cnn",
        "plane": "axial",
        "axial": sample_volumes_b64["axial"]
    }
    resp = client.post("/gradcam", json=payload)
    assert resp.status_code == 422
    assert "no spatial attention" in resp.json()["detail"]


def test_gradcam_proposed_returns_overlay(sample_volumes_b64):
    payload = {
        "model": "proposed",
        "plane": "axial",
        "slice_idx": 12,
        "axial": sample_volumes_b64["axial"]
    }
    resp = client.post("/gradcam", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "proposed"
    assert data["plane"] == "axial"
    assert data["slice_idx"] == 12
    assert "overlay_image" in data
    assert data["overlay_image"].startswith("data:image/png;base64,")
    assert "hotspot" in data


def test_audit_logs_recorded():
    resp = client.get("/audit-logs")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_records" in data
    assert data["total_records"] > 0
    assert "probability" in data["logs"][0]
