"""
Golden-Case Regression Test.
Validates that inference on held-out test cases (Exams 1130 and 1172) reproduces the
exact notebook ground truth and test_logits_acl.csv values.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

API_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(API_DIR))

from inference import engine

BASE_DIR = API_DIR.parent
DATASET_VALID_DIR = BASE_DIR / "dataset" / "valid"


@pytest.fixture(scope="module", autouse=True)
def setup_engine():
    engine.load_all()


def load_exam_volumes(exam_id: str):
    vols = {}
    for plane in ["axial", "coronal", "sagittal"]:
        p = DATASET_VALID_DIR / plane / f"{exam_id}.npy"
        assert p.exists(), f"Missing {p}"
        vols[plane] = np.load(p)
    return vols["axial"], vols["coronal"], vols["sagittal"]


def test_golden_case_1130_normal():
    """
    Exam 1130: Held-out test set normal/intact knee (Ground truth: 0.0).
    Notebook p_test: 0.000054 (0.005%).
    Expected prediction: 'normal'.
    """
    axial, coronal, sagittal = load_exam_volumes("1130")

    res = engine.predict(
        model_key="proposed",
        axial_vol=axial,
        coronal_vol=coronal,
        sagittal_vol=sagittal,
        threshold=0.5
    )

    assert res["model"] == "proposed"
    assert res["predicted_label"] == "normal"
    # Expected probability ~0.000054, assert strictly < 0.01 (clear normal)
    assert res["probability"] < 0.01, f"Expected near-zero probability, got {res['probability']}"

    # Verify axial logit ~ -5.039 (CPU fp32 vs GPU fp16 autocast difference < 0.05)
    assert np.isclose(res["plane_logits"]["axial"], -5.039, atol=0.05), (
        f"Axial logit mismatch: got {res['plane_logits']['axial']}, expected ~-5.039"
    )


def test_golden_case_1172_acl_tear():
    """
    Exam 1172: Held-out test set positive ACL tear (Ground truth: 1.0).
    Notebook p_test: 0.930790 (93.1%).
    Expected prediction: 'tear'.
    """
    axial, coronal, sagittal = load_exam_volumes("1172")

    res = engine.predict(
        model_key="proposed",
        axial_vol=axial,
        coronal_vol=coronal,
        sagittal_vol=sagittal,
        threshold=0.5
    )

    assert res["model"] == "proposed"
    assert res["predicted_label"] == "tear"
    # Expected probability ~0.9308, assert > 0.85
    assert res["probability"] > 0.85, f"Expected high tear probability, got {res['probability']}"

    # Verify axial logit ~ 1.737
    assert np.isclose(res["plane_logits"]["axial"], 1.737, atol=0.05), (
        f"Axial logit mismatch: got {res['plane_logits']['axial']}, expected ~1.737"
    )


def test_compare_endpoint_runs_all_models():
    """
    Tests that predict_compare returns results for proposed, transfer_learning, and custom_cnn.
    """
    axial, coronal, sagittal = load_exam_volumes("1130")
    results = engine.predict_compare(
        axial_vol=axial,
        coronal_vol=coronal,
        sagittal_vol=sagittal,
        threshold=0.5
    )

    assert len(results) == 3
    models_returned = [r["model"] for r in results]
    assert models_returned == ["proposed", "transfer_learning", "custom_cnn"]

    # All three should classify exam 1130 as normal
    for r in results:
        assert r["predicted_label"] == "normal"
