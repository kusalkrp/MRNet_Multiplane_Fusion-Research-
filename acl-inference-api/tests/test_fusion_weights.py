"""
Test that /predict fusion weights match clf.coef_ from the saved joblib files exactly.
"""

import pytest
import joblib
import numpy as np
from pathlib import Path
import sys

# Ensure acl-inference-api is in sys.path
API_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(API_DIR))

from inference import engine, MODEL_REGISTRY


@pytest.fixture(scope="module", autouse=True)
def setup_engine():
    engine.load_all()


def test_fusion_weights_exact_match():
    """Verify that get_fusion_weights matches clf.coef_ from joblib for all 3 models."""
    for model_key, spec in MODEL_REGISTRY.items():
        joblib_path = engine.checkpoints_dir / spec["fusion_filename"]
        pipeline = joblib.load(joblib_path)
        clf = pipeline.named_steps["clf"]

        expected_coef = clf.coef_[0]
        expected_intercept = float(clf.intercept_[0])

        weights = engine.get_fusion_weights(model_key)

        assert np.isclose(weights["axial"], expected_coef[0], atol=1e-6), f"Axial coef mismatch for {model_key}"
        assert np.isclose(weights["coronal"], expected_coef[1], atol=1e-6), f"Coronal coef mismatch for {model_key}"
        assert np.isclose(weights["sagittal"], expected_coef[2], atol=1e-6), f"Sagittal coef mismatch for {model_key}"
        assert np.isclose(weights["bias"], expected_intercept, atol=1e-6), f"Bias mismatch for {model_key}"


def test_fusion_weights_distinct_across_models():
    """Verify that all three models have distinct fusion weights (no accidental duplicates)."""
    weights_hyb = engine.get_fusion_weights("proposed")
    weights_cnn = engine.get_fusion_weights("custom_cnn")
    weights_tra = engine.get_fusion_weights("transfer_learning")

    # Coef axial
    assert weights_hyb["axial"] != weights_cnn["axial"]
    assert weights_hyb["axial"] != weights_tra["axial"]
    assert weights_cnn["axial"] != weights_tra["axial"]

    # Biases
    assert weights_hyb["bias"] != weights_cnn["bias"]
    assert weights_hyb["bias"] != weights_tra["bias"]
