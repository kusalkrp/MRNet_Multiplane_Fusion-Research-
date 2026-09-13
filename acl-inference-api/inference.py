"""
Inference Engine and Model Registry.
Caches 9 PyTorch Plane Experts and 3 Stage-2 Logistic Regression fusion pipelines in memory.
Executes single-model and multi-model comparison predictions.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import joblib
import torch
import torch.nn as nn

try:
    from .models.hybrid_expert import HybridPlaneExpert
    from .models.custom_cnn_expert import CustomCNNPlaneExpert
    from .models.transfer_expert import TransferLearningPlaneExpert
    from .preprocessing import preprocess_volume_to_tensor
except (ImportError, ValueError):
    from models.hybrid_expert import HybridPlaneExpert
    from models.custom_cnn_expert import CustomCNNPlaneExpert
    from models.transfer_expert import TransferLearningPlaneExpert
    from preprocessing import preprocess_volume_to_tensor

# Base directories
PACKAGE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"

PLANES = ["axial", "coronal", "sagittal"]

# Device configuration (CPU or CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REGISTRY = {
    "proposed": {
        "name": "Proposed Hybrid Expert",
        "description": "DenseNet121 + Custom CNN + CBAM Attention + Slice Attention Fusion",
        "class": HybridPlaneExpert,
        "checkpoint_pattern": "hybrid_expert_acl_{plane}.pt",
        "fusion_filename": "fusion_acl_hybrid.joblib",
        "has_attention": True,
        "paper_auc": 0.954,
    },
    "transfer_learning": {
        "name": "Transfer Learning Expert",
        "description": "Pretrained DenseNet121 Backbone + Global Max Pooling",
        "class": TransferLearningPlaneExpert,
        "checkpoint_pattern": "transfer_expert_acl_{plane}.pt",
        "fusion_filename": "fusion_acl_transfer.joblib",
        "has_attention": False,
        "paper_auc": 0.892,
    },
    "custom_cnn": {
        "name": "Custom CNN Expert",
        "description": "4-Block Custom CNN (Trained from Scratch) + Global Max Pooling",
        "class": CustomCNNPlaneExpert,
        "checkpoint_pattern": "custom_cnn_expert_acl_{plane}.pt",
        "fusion_filename": "fusion_acl_custom_cnn.joblib",
        "has_attention": False,
        "paper_auc": 0.827,
    },
}


class ModelNotAvailableError(Exception):
    """Raised when an expert model or checkpoint failed to load."""
    pass


class ModelNotFoundError(Exception):
    """Raised when an unknown model key is requested."""
    pass


class InferenceEngine:
    """
    Central inference manager:
    - Preloads and holds all 9 PyTorch models and 3 fusion pipelines in memory.
    - Manages tensor forward passes and probability fusion.
    """
    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR, device: torch.device = DEVICE):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device
        self.models: Dict[str, Dict[str, nn.Module]] = {}
        self.fusion_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.load_status: Dict[str, bool] = {}

    def load_all(self):
        """Loads all registered models and fusion models at startup."""
        for model_key, spec in MODEL_REGISTRY.items():
            self.models[model_key] = {}
            plane_metrics = {}
            all_planes_ok = True

            # 1. Load the 3 plane expert PyTorch models
            for plane in PLANES:
                ckpt_filename = spec["checkpoint_pattern"].format(plane=plane)
                ckpt_path = self.checkpoints_dir / ckpt_filename

                if not ckpt_path.exists():
                    all_planes_ok = False
                    print(f"Warning: Checkpoint missing for {model_key} {plane}: {ckpt_path}")
                    continue

                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    config = ckpt.get("config", {})
                    model_cls = spec["class"]
                    model = model_cls(config)
                    model.load_state_dict(ckpt["model_state"], strict=True)
                    model.to(self.device)
                    model.eval()

                    self.models[model_key][plane] = model
                    plane_metrics[plane] = {
                        "best_auc": float(ckpt.get("best_auc", float("nan"))),
                        "loss_type": str(ckpt.get("loss_type", "bce")),
                        "checkpoint": ckpt_filename
                    }
                except Exception as e:
                    all_planes_ok = False
                    print(f"Error loading {model_key} {plane} from {ckpt_path}: {e}")

            # 2. Load the Stage-2 Fusion Pipeline
            fusion_path = self.checkpoints_dir / spec["fusion_filename"]
            if fusion_path.exists():
                try:
                    pipeline = joblib.load(fusion_path)
                    self.fusion_models[model_key] = pipeline
                except Exception as e:
                    all_planes_ok = False
                    print(f"Error loading fusion pipeline for {model_key} from {fusion_path}: {e}")
            else:
                all_planes_ok = False
                print(f"Warning: Fusion model missing for {model_key}: {fusion_path}")

            self.load_status[model_key] = all_planes_ok
            self.model_metadata[model_key] = {
                "name": spec["name"],
                "description": spec["description"],
                "has_attention": spec["has_attention"],
                "paper_auc": spec["paper_auc"],
                "is_loaded": all_planes_ok,
                "planes": plane_metrics,
            }

    def get_model(self, model_key: str, plane: str) -> nn.Module:
        if model_key not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Unknown model key '{model_key}'. Must be one of {list(MODEL_REGISTRY.keys())}")
        if not self.load_status.get(model_key, False) or plane not in self.models.get(model_key, {}):
            raise ModelNotAvailableError(f"Model '{model_key}' plane '{plane}' is not loaded/available.")
        return self.models[model_key][plane]

    def get_fusion_model(self, model_key: str):
        if model_key not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Unknown model key '{model_key}'.")
        if model_key not in self.fusion_models:
            raise ModelNotAvailableError(f"Fusion model for '{model_key}' is not available.")
        return self.fusion_models[model_key]

    def get_fusion_weights(self, model_key: str) -> Dict[str, float]:
        """Extracts linear coefficients and bias from Stage-2 Logistic Regression."""
        pipeline = self.get_fusion_model(model_key)
        clf = pipeline.named_steps["clf"]
        coef = clf.coef_[0]
        intercept = clf.intercept_[0]
        return {
            "axial": float(coef[0]),
            "coronal": float(coef[1]),
            "sagittal": float(coef[2]),
            "bias": float(intercept),
        }

    def predict_plane(self, model_key: str, plane: str, volume: np.ndarray) -> float:
        """Executes forward pass for a single plane volume, returns float logit."""
        model = self.get_model(model_key, plane)
        x_tensor = preprocess_volume_to_tensor(volume, plane_name=plane).to(self.device)

        with torch.no_grad():
            logit = model(x_tensor)
            if isinstance(logit, torch.Tensor):
                val = float(logit.item()) if logit.numel() == 1 else float(logit[0].item())
            else:
                val = float(logit)
        return val

    def predict(
        self,
        model_key: str,
        axial_vol: np.ndarray,
        coronal_vol: np.ndarray,
        sagittal_vol: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Runs multiplane inference for a specific model key:
        1. Axial, coronal, sagittal forward passes -> 3 plane logits.
        2. LogisticRegression fusion -> fused probability & classification.
        3. Returns full dictionary matching API specification.
        """
        if model_key not in MODEL_REGISTRY:
            raise ModelNotFoundError(f"Unknown model '{model_key}'. Allowed models: {list(MODEL_REGISTRY.keys())}")

        if not self.load_status.get(model_key, False):
            raise ModelNotAvailableError(f"Model '{model_key}' is not fully loaded on this server.")

        # Compute logits for all 3 planes
        logit_axial = self.predict_plane(model_key, "axial", axial_vol)
        logit_coronal = self.predict_plane(model_key, "coronal", coronal_vol)
        logit_sagittal = self.predict_plane(model_key, "sagittal", sagittal_vol)

        plane_logits = {
            "axial": round(logit_axial, 4),
            "coronal": round(logit_coronal, 4),
            "sagittal": round(logit_sagittal, 4),
        }

        # Stage 2 Fusion prediction
        fusion_pipeline = self.get_fusion_model(model_key)
        logits_array = np.array([[logit_axial, logit_coronal, logit_sagittal]])
        prob = float(fusion_pipeline.predict_proba(logits_array)[0, 1])

        predicted_label = "tear" if prob >= threshold else "normal"
        fusion_weights = self.get_fusion_weights(model_key)
        has_attention = MODEL_REGISTRY[model_key]["has_attention"]

        return {
            "model": model_key,
            "probability": round(prob, 4),
            "predicted_label": predicted_label,
            "threshold": threshold,
            "plane_logits": plane_logits,
            "fusion_weights": {k: round(v, 4) for k, v in fusion_weights.items()},
            "has_attention": has_attention,
        }

    def predict_compare(
        self,
        axial_vol: np.ndarray,
        coronal_vol: np.ndarray,
        sagittal_vol: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Runs inference across all registered models and returns a list of results."""
        results = []
        for model_key in ["proposed", "transfer_learning", "custom_cnn"]:
            if self.load_status.get(model_key, False):
                res = self.predict(
                    model_key=model_key,
                    axial_vol=axial_vol,
                    coronal_vol=coronal_vol,
                    sagittal_vol=sagittal_vol,
                    threshold=threshold
                )
                results.append(res)
        return results


# Global singleton instance
engine = InferenceEngine()
