"""
Grad-CAM Explainability Module for TriPlane Knee MRI.
Ported directly from research notebook visualization cells:
- HybridPlaneExpertGradCAM wrapper for slice-level gradient computation.
- Generates Grad-CAM activation maps, colored JET/Turbo heatmaps, and alpha-blended overlays.
- Serializes images to base64 Data URLs for web client consumption.
"""

import io
import base64
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

try:
    from .models.hybrid_expert import HybridPlaneExpert
    from .models.transfer_expert import TransferLearningPlaneExpert
    from .preprocessing import (
        intensity_normalize,
        standardize_volume,
        preprocess_slice_to_tensor
    )
except (ImportError, ValueError):
    from models.hybrid_expert import HybridPlaneExpert
    from models.transfer_expert import TransferLearningPlaneExpert
    from preprocessing import (
        intensity_normalize,
        standardize_volume,
        preprocess_slice_to_tensor
    )


class GradCAMNotSupportedError(Exception):
    """Raised when Grad-CAM is requested for a model that does not support it (e.g. custom_cnn)."""
    pass


class HybridPlaneExpertGradCAM(nn.Module):
    """
    Wrapper for HybridPlaneExpert and TransferLearningPlaneExpert to enable Grad-CAM visualization.
    Processes a single slice while exposing target layer activations and gradients.
    """
    def __init__(self, expert_model: nn.Module, target_layer_name: str = "cbam"):
        super().__init__()
        self.expert = expert_model
        self.target_layer_name = target_layer_name

        if isinstance(expert_model, HybridPlaneExpert):
            if target_layer_name == "cbam" and hasattr(expert_model, "cbam"):
                self.target_layer = expert_model.cbam
            elif target_layer_name == "backbone":
                if hasattr(expert_model.backbone, "features"):
                    self.target_layer = expert_model.backbone.features[-1]
                else:
                    self.target_layer = expert_model.backbone[-1]
            elif hasattr(expert_model, "cbam"):
                self.target_layer = expert_model.cbam
            else:
                self.target_layer = expert_model.backbone.features[-1]
        elif isinstance(expert_model, TransferLearningPlaneExpert):
            if hasattr(expert_model.backbone, "features"):
                self.target_layer = expert_model.backbone.features[-1]
            else:
                self.target_layer = expert_model.backbone[-1]
        else:
            raise GradCAMNotSupportedError(f"Model architecture {type(expert_model).__name__} does not support Grad-CAM.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single slice: adds depth dimension (B, 1, C, H, W)."""
        x_volume = x.unsqueeze(1)
        return self.expert(x_volume)

    def get_target_layer(self):
        return self.target_layer


def pil_to_base64_url(img: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL Image into a base64 Data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def generate_gradcam_for_volume(
    model: nn.Module,
    model_key: str,
    plane: str,
    volume: np.ndarray,
    slice_idx: Optional[int] = None,
    target_layer_name: str = "cbam",
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Generates Grad-CAM visual explainability for an MRI volume and specified slice.
    """
    if model_key == "custom_cnn":
        raise GradCAMNotSupportedError(
            "Grad-CAM is not supported for 'custom_cnn'. The Custom CNN baseline lacks attention mechanisms and pretrained spatial feature resolution."
        )

    if not isinstance(model, (HybridPlaneExpert, TransferLearningPlaneExpert)):
        raise GradCAMNotSupportedError(f"Model key '{model_key}' does not support Grad-CAM visualization.")

    # 1. Volume standardization & slice selection
    volume_norm = intensity_normalize(volume)
    volume_std = standardize_volume(volume_norm, target_slices=25)
    total_slices = volume_std.shape[0]

    if slice_idx is None:
        slice_idx = total_slices // 2
    else:
        slice_idx = max(0, min(int(slice_idx), total_slices - 1))

    raw_slice_2d = volume_std[slice_idx]

    # 2. Transform target slice to tensor
    slice_tensor = preprocess_slice_to_tensor(raw_slice_2d, img_size=224).to(device)
    input_batch = slice_tensor.unsqueeze(0)  # (1, 3, 224, 224)

    # 3. Initialize wrapper and Grad-CAM
    wrapper = HybridPlaneExpertGradCAM(model, target_layer_name=target_layer_name).to(device)
    target_layer = wrapper.get_target_layer()

    cam = GradCAM(model=wrapper, target_layers=[target_layer])
    targets = [BinaryClassifierOutputTarget(1)]

    # Compute CAM
    grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0]

    # Predict score
    with torch.no_grad():
        logit = wrapper(input_batch)
        val = float(logit.item()) if logit.numel() == 1 else float(logit[0].item())
        confidence = float(torch.sigmoid(torch.tensor(val)).item())

    # 4. Prepare images for visualization
    s_min, s_max = float(raw_slice_2d.min()), float(raw_slice_2d.max())
    denom = s_max - s_min if s_max - s_min > 0 else 1.0
    slice_uint8 = ((raw_slice_2d - s_min) / denom * 255).astype(np.uint8)
    slice_rgb = cv2.cvtColor(cv2.resize(slice_uint8, (224, 224)), cv2.COLOR_GRAY2RGB)

    img_float = slice_rgb.astype(np.float32) / 255.0

    # Create JET / Turbo heatmap
    cam_heatmap_bgr = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    cam_heatmap_rgb = cv2.cvtColor(cam_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay with 50% blend
    cam_overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.55)

    # Hotspot detection (peak attention point)
    max_y, max_x = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)

    # PIL conversion and base64 encoding
    orig_pil = Image.fromarray(slice_rgb)
    heat_pil = Image.fromarray(cam_heatmap_rgb)
    over_pil = Image.fromarray(cam_overlay)

    return {
        "model": model_key,
        "plane": plane,
        "slice_idx": slice_idx,
        "total_slices": total_slices,
        "confidence": round(confidence, 4),
        "target_layer": target_layer_name,
        "hotspot": {"x": int(max_x), "y": int(max_y)},
        "original_image": pil_to_base64_url(orig_pil),
        "heatmap_image": pil_to_base64_url(heat_pil),
        "overlay_image": pil_to_base64_url(over_pil),
    }
