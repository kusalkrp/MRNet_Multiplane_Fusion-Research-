"""
PyTorch Real-Time Inference Engine for Knee MRI Injury Detection
Supports Hybrid (Proposed), Custom CNN, and Transfer Learning models.
Includes Volume Preprocessing, Multi-Plane Stage 2 Fusion, and Grad-CAM Generation.
"""

import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import joblib
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import (
    alexnet, AlexNet_Weights,
    densenet121, DenseNet121_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
import streamlit as st

# Path definitions
BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "Code"
if not CODE_DIR.exists():
    CODE_DIR = BASE_DIR / "MRNet Hybrid"

DATASET_DIR = BASE_DIR / "dataset"
if not DATASET_DIR.exists():
    DATASET_DIR = CODE_DIR / "dataset"

# Model folders
HYBRID_DIR = CODE_DIR / "runs_mrnet_hybrid_fusion_npy"
CNN_ONLY_DIR = CODE_DIR / "runs_mrnet_custom_cnn_only"
TRANSFER_DIR = CODE_DIR / "runs_mrnet_transfer_learning_only"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. MODEL ARCHITECTURE DEFINITIONS
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention: Learns WHICH feature channels are important."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention: Learns WHERE to focus in the image."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(combined))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CustomCNN(nn.Module):
    """Custom 4-block CNN for MRI feature extraction."""
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SliceAttention(nn.Module):
    """Attention-based slice aggregation."""
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, slice_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_scores = self.attention(slice_features)
        attn_weights = F.softmax(attn_scores, dim=0)
        aggregated = (slice_features * attn_weights).sum(dim=0)
        return aggregated, attn_weights.squeeze()


def get_backbone(backbone_name: str, pretrained: bool = False):
    """Factory for backbone feature extractors."""
    if backbone_name == "alexnet":
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        base = alexnet(weights=weights)
        features = base.features
        num_features = 256
    elif backbone_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base = densenet121(weights=weights)
        features = base.features
        num_features = 1024
    elif backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        features = nn.Sequential(*list(base.children())[:-2])
        num_features = 512
    elif backbone_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)
        features = base.features
        num_features = 1280
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return features, num_features


class HybridPlaneExpert(nn.Module):
    """
    Hybrid Enhanced Plane Expert (Backbone + Custom CNN + CBAM + Slice Attention).
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        backbone_name = str(config.get("backbone", "densenet121"))
        feature_dim = int(config.get("feature_dim", 256))
        
        self.backbone, self.backbone_features = get_backbone(backbone_name, pretrained=False)
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        
        if config.get("use_cbam", True):
            self.cbam = CBAM(self.backbone_features, reduction=16)
        
        if config.get("use_custom_cnn", True):
            self.custom_cnn = CustomCNN(out_features=feature_dim)
            combined_dim = self.backbone_features + feature_dim
        else:
            combined_dim = self.backbone_features
        
        self.combined_dim = combined_dim
        
        if config.get("use_slice_attention", True):
            self.slice_attention = SliceAttention(combined_dim, hidden_dim=64)
        
        dropout = float(config.get("dropout", 0.4))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def encode_slice(self, x: torch.Tensor) -> torch.Tensor:
        backbone_feat = self.backbone(x)
        if self.config.get("backbone") != "alexnet":
            backbone_feat = F.relu(backbone_feat)
        
        if self.config.get("use_cbam", True) and hasattr(self, 'cbam'):
            backbone_feat = self.cbam(backbone_feat)
        
        backbone_feat = self.backbone_pool(backbone_feat).flatten(1)
        
        if self.config.get("use_custom_cnn", True) and hasattr(self, 'custom_cnn'):
            custom_feat = self.custom_cnn(x)
            combined = torch.cat([backbone_feat, custom_feat], dim=1)
        else:
            combined = backbone_feat
        
        return combined
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = x.shape
        x_flat = x.view(b * k, c, h, w)
        
        slice_features = self.encode_slice(x_flat)
        slice_features = slice_features.view(b, k, -1)
        
        if self.config.get("use_slice_attention", True) and hasattr(self, 'slice_attention'):
            aggregated = []
            for i in range(b):
                agg, _ = self.slice_attention(slice_features[i])
                aggregated.append(agg)
            v = torch.stack(aggregated, dim=0)
        else:
            v = torch.amax(slice_features, dim=1)
        
        logit = self.head(v).squeeze(1)
        return logit


class CustomCNNPlaneExpert(nn.Module):
    """Custom CNN Only Plane Expert (Trained from scratch, max pooling)."""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        feature_dim = int(config.get("feature_dim", 256))
        
        self.custom_cnn = CustomCNN(out_features=feature_dim)
        dropout = float(config.get("dropout", 0.4))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def encode_slice(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_cnn(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = x.shape
        x_flat = x.view(b * k, c, h, w)
        slice_features = self.encode_slice(x_flat).view(b, k, -1)
        v = torch.amax(slice_features, dim=1)
        logit = self.head(v).squeeze(1)
        return logit


class TransferLearningPlaneExpert(nn.Module):
    """Transfer Learning Only Plane Expert (Pretrained backbone only)."""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        backbone_name = str(config.get("backbone", "densenet121"))
        self.backbone, self.backbone_features = get_backbone(backbone_name, pretrained=False)
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)
        
        dropout = float(config.get("dropout", 0.4))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def encode_slice(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self.config.get("backbone") != "alexnet":
            feat = F.relu(feat)
        feat = self.backbone_pool(feat).flatten(1)
        return feat
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = x.shape
        x_flat = x.view(b * k, c, h, w)
        slice_features = self.encode_slice(x_flat).view(b, k, -1)
        v = torch.amax(slice_features, dim=1)
        logit = self.head(v).squeeze(1)
        return logit


# ============================================================================
# 2. PREPROCESSING & TRANSFORMATIONS
# ============================================================================

def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    v_min, v_max = volume.min(), volume.max()
    if v_max - v_min > 0:
        return (volume - v_min) / (v_max - v_min)
    return volume

def z_score_normalize(volume: np.ndarray) -> np.ndarray:
    mean, std = volume.mean(), volume.std()
    if std > 0:
        return (volume - mean) / std
    return volume - mean

def intensity_normalize(volume: np.ndarray) -> np.ndarray:
    volume = min_max_normalize(volume)
    volume = z_score_normalize(volume)
    return volume

def calculate_slice_entropy(slice_2d: np.ndarray) -> float:
    s_min, s_max = slice_2d.min(), slice_2d.max()
    denom = (s_max - s_min + 1e-8)
    slice_norm = ((slice_2d - s_min) / denom * 255).astype(np.uint8)
    hist, _ = np.histogram(slice_norm.flatten(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(entropy(hist))

def hybrid_slice_selection(volume: np.ndarray, target_slices: int = 25, center_ratio: float = 0.4) -> np.ndarray:
    n_slices = volume.shape[0]
    if n_slices <= target_slices:
        return volume
    
    n_center = int(target_slices * center_ratio)
    n_entropy = target_slices - n_center
    
    center_start = (n_slices - n_center) // 2
    center_end = center_start + n_center
    center_indices = list(range(center_start, center_end))
    
    peripheral_indices = list(range(0, center_start)) + list(range(center_end, n_slices))
    
    if len(peripheral_indices) > 0 and n_entropy > 0:
        entropies = np.array([calculate_slice_entropy(volume[i]) for i in peripheral_indices])
        if len(peripheral_indices) >= n_entropy:
            top_entropy_local = np.argsort(entropies)[-n_entropy:]
            entropy_indices = [peripheral_indices[i] for i in top_entropy_local]
        else:
            entropy_indices = peripheral_indices
    else:
        entropy_indices = []
    
    all_indices = sorted(set(center_indices + entropy_indices))
    
    if len(all_indices) > target_slices:
        step = len(all_indices) / target_slices
        all_indices = [all_indices[int(i * step)] for i in range(target_slices)]
    
    while len(all_indices) < target_slices:
        remaining = [i for i in range(n_slices) if i not in all_indices]
        if not remaining:
            break
        entropies_remaining = [calculate_slice_entropy(volume[i]) for i in remaining]
        best_idx = remaining[np.argmax(entropies_remaining)]
        all_indices.append(best_idx)
        all_indices = sorted(all_indices)
    
    return volume[all_indices[:target_slices]]

def standardize_volume(volume: np.ndarray, target_slices: int = 25) -> np.ndarray:
    n_slices = volume.shape[0]
    if n_slices == target_slices:
        return volume
    elif n_slices < target_slices:
        pad_total = target_slices - n_slices
        pad_before = pad_total // 2
        padded = np.zeros((target_slices, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
        padded[pad_before:pad_before + n_slices] = volume
        return padded
    else:
        return hybrid_slice_selection(volume, target_slices=target_slices)

def preprocess_slice_to_tensor(slice_2d: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Converts 2D slice to 3-channel normalized ImageNet float tensor (3, H, W)."""
    # Min-max scaling to [0, 1] if raw integer/float
    s_min, s_max = slice_2d.min(), slice_2d.max()
    if s_max - s_min > 0:
        slice_2d = (slice_2d - s_min) / (s_max - s_min)
    
    x = torch.from_numpy(slice_2d).float().unsqueeze(0)  # (1, H, W)
    x = F.interpolate(x.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(0)
    x = x.repeat(3, 1, 1)  # (3, H, W)
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(x)

def preprocess_volume_to_tensor(volume: np.ndarray, target_slices: int = 25, img_size: int = 224) -> torch.Tensor:
    """Preprocesses a 3D volume (S, H, W) into (1, 25, 3, 224, 224) PyTorch batch tensor."""
    volume = intensity_normalize(volume)
    volume = standardize_volume(volume, target_slices=target_slices)
    
    slice_tensors = [preprocess_slice_to_tensor(volume[s], img_size) for s in range(volume.shape[0])]
    volume_tensor = torch.stack(slice_tensors, dim=0)  # (K, 3, H, W)
    return volume_tensor.unsqueeze(0)  # (1, K, 3, H, W)


# ============================================================================
# 3. CACHED MODEL & FUSION LOADERS
# ============================================================================

@st.cache_resource
def load_expert_model(model_key: str, task: str, plane: str) -> Tuple[Optional[nn.Module], Dict]:
    """
    Loads trained PyTorch expert model based on model_key, task, and plane.
    model_key: 'hybrid' | 'custom_cnn' | 'transfer_learning'
    task: 'acl' | 'meniscus' | 'abnormal'
    plane: 'axial' | 'coronal' | 'sagittal'
    """
    dir_map = {
        "hybrid": (HYBRID_DIR, "hybrid_expert"),
        "custom_cnn": (CNN_ONLY_DIR, "customcnn_expert"),
        "transfer_learning": (TRANSFER_DIR, "transferlearning_expert")
    }
    
    if model_key not in dir_map:
        return None, {"error": f"Unknown model key: {model_key}"}
    
    folder, prefix = dir_map[model_key]
    ckpt_path = folder / f"{prefix}_{task}_{plane}.pt"
    
    if not ckpt_path.exists():
        # Fallback check for alternate names
        alt_path = folder / f"{prefix}_{task}.pt"
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            return None, {"error": f"Checkpoint not found: {ckpt_path.name}"}
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt.get("config", {})
        
        if model_key == "hybrid":
            model = HybridPlaneExpert(config)
        elif model_key == "custom_cnn":
            model = CustomCNNPlaneExpert(config)
        else:
            model = TransferLearningPlaneExpert(config)
        
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(DEVICE)
        model.eval()
        
        info = {
            "best_auc": ckpt.get("best_auc", float("nan")),
            "plane": plane,
            "task": task,
            "path": str(ckpt_path),
            "config": config
        }
        return model, info
    except Exception as e:
        return None, {"error": f"Failed to load checkpoint: {str(e)}"}


@st.cache_resource
def load_fusion_model(model_key: str, task: str):
    """Loads Stage 2 Logistic Regression fusion classifier."""
    dir_map = {
        "hybrid": HYBRID_DIR,
        "custom_cnn": CNN_ONLY_DIR,
        "transfer_learning": TRANSFER_DIR
    }
    folder = dir_map.get(model_key, HYBRID_DIR)
    fusion_path = folder / f"fusion_{task}.joblib"
    
    if fusion_path.exists():
        try:
            model = joblib.load(fusion_path)
            return model
        except Exception:
            pass
    return None


# ============================================================================
# 4. INFERENCE PIPELINE EXECUTION
# ============================================================================

def run_single_plane_inference(model_key: str, task: str, plane: str, volume: np.ndarray) -> Dict:
    """Runs inference for a single MRI plane volume."""
    model, info = load_expert_model(model_key, task, plane)
    if model is None:
        return {"error": info.get("error", "Model load error")}
    
    tensor = preprocess_volume_to_tensor(volume).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor).item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
    
    return {
        "plane": plane,
        "logit": logit,
        "prob": prob,
        "best_auc": info.get("best_auc", float("nan"))
    }


def run_multiplane_inference(
    model_key: str, 
    task: str, 
    axial_vol: np.ndarray, 
    coronal_vol: np.ndarray, 
    sagittal_vol: np.ndarray
) -> Dict:
    """
    Full end-to-end multiplane prediction:
    1. axial, coronal, sagittal forward passes -> 3 plane logits
    2. Stage 2 Logistic Regression fusion -> fused probability & final diagnosis
    """
    vols = {
        "axial": axial_vol,
        "coronal": coronal_vol,
        "sagittal": sagittal_vol
    }
    
    plane_results = {}
    logits = []
    
    for plane in ["axial", "coronal", "sagittal"]:
        res = run_single_plane_inference(model_key, task, plane, vols[plane])
        if "error" in res:
            return {"error": f"Error in {plane} plane: {res['error']}"}
        plane_results[plane] = res
        logits.append(res["logit"])
    
    # Stage 2 Fusion
    fusion_clf = load_fusion_model(model_key, task)
    if fusion_clf is not None:
        fused_prob = float(fusion_clf.predict_proba([logits])[0][1])
    else:
        # Fallback average probability if joblib missing
        probs = [res["prob"] for res in plane_results.values()]
        fused_prob = float(np.mean(probs))
    
    is_positive = fused_prob >= 0.5
    prediction_label = "Positive (Injury Detected)" if is_positive else "Negative (Normal)"
    confidence = fused_prob if is_positive else (1 - fused_prob)
    
    return {
        "task": task,
        "model_key": model_key,
        "plane_results": plane_results,
        "logits": {
            "axial": logits[0],
            "coronal": logits[1],
            "sagittal": logits[2]
        },
        "probs": {
            "axial": plane_results["axial"]["prob"],
            "coronal": plane_results["coronal"]["prob"],
            "sagittal": plane_results["sagittal"]["prob"]
        },
        "fused_prob": fused_prob,
        "prediction_label": prediction_label,
        "confidence": confidence * 100
    }


# ============================================================================
# 5. REAL-TIME GRAD-CAM VISUALIZATION
# ============================================================================

def generate_gradcam_heatmap(
    model_key: str,
    task: str,
    plane: str,
    volume: np.ndarray,
    target_slice_idx: Optional[int] = None
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
    """
    Generates real-time Grad-CAM heatmap for a target slice in the volume.
    Returns: (original_slice_img, heatmap_img, overlay_img)
    """
    model, _ = load_expert_model(model_key, task, plane)
    if model is None:
        return None, None, None
    
    # Preprocess volume to tensor
    volume_norm = intensity_normalize(volume)
    volume_std = standardize_volume(volume_norm, target_slices=25)
    
    if target_slice_idx is None:
        target_slice_idx = volume_std.shape[0] // 2  # default center slice
    
    target_slice_idx = max(0, min(target_slice_idx, volume_std.shape[0] - 1))
    raw_slice_2d = volume_std[target_slice_idx]
    
    # Prepare input tensor with autograd enabled
    volume_tensor = preprocess_volume_to_tensor(volume_std).to(DEVICE)
    volume_tensor.requires_grad_(True)
    
    # Identify target layer (last Conv2d layer) for Grad-CAM
    target_layer = None
    if model_key == "hybrid" and hasattr(model, "custom_cnn"):
        for m in model.custom_cnn.modules():
            if isinstance(m, nn.Conv2d):
                target_layer = m
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                target_layer = m
    
    if target_layer is None:
        return None, None, None
    
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        model.zero_grad()
        logit = model(volume_tensor)
        # Backpropagate sigmoid probability score for stable positive/negative gradient flow
        score = torch.sigmoid(logit).squeeze()
        score.backward()
        
        h_fwd.remove()
        h_bwd.remove()
        
        if not activations or not gradients:
            return None, None, None
        
        act_batch = activations[0].detach().cpu().numpy()  # (K, C, H, W)
        grad_batch = gradients[0].detach().cpu().numpy()   # (K, C, H, W)
        
        target_idx = max(0, min(target_slice_idx, act_batch.shape[0] - 1))
        act = act_batch[target_idx]    # (C, H, W)
        grad = grad_batch[target_idx]  # (C, H, W)
        
        # Mean gradients across spatial dimensions
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]
        
        # Relative min-max normalization to highlight attention regions regardless of logit sign
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min > 1e-8:
            cam_norm = (cam - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam)
        
        # Resize CAM to 224x224
        if cv2 is not None:
            cam_resized = cv2.resize(cam_norm, (224, 224))
        else:
            cam_pil = Image.fromarray((cam_norm * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0
        
        # Prepare original slice PIL Image
        s_min, s_max = raw_slice_2d.min(), raw_slice_2d.max()
        denom = s_max - s_min if s_max - s_min > 0 else 1
        img_gray = ((raw_slice_2d - s_min) / denom * 255).astype(np.uint8)
        
        if cv2 is not None:
            img_gray_resized = cv2.resize(img_gray, (224, 224))
            img_rgb = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2RGB)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        else:
            img_gray_pil = Image.fromarray(img_gray).resize((224, 224), Image.BILINEAR)
            img_rgb = np.stack([np.array(img_gray_pil)] * 3, axis=-1)
            cmap = plt.get_cmap('jet')
            heatmap_rgb = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
        
        # Dynamic intensity-based alpha blending: attention areas light up in red/yellow, low areas retain grayscale MRI
        alpha = (cam_resized * 0.70)[:, :, np.newaxis]
        overlay = (img_rgb * (1.0 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        
        orig_pil = Image.fromarray(img_rgb)
        heat_pil = Image.fromarray(heatmap_rgb)
        over_pil = Image.fromarray(overlay)
        
        return orig_pil, heat_pil, over_pil
    except Exception as e:
        if 'h_fwd' in locals(): h_fwd.remove()
        if 'h_bwd' in locals(): h_bwd.remove()
        return None, None, None


# ============================================================================
# 6. DATASET VALIDATION SAMPLES HELPER
# ============================================================================

def get_available_sample_ids() -> List[str]:
    """Returns sorted list of test/validation exam IDs available in the dataset."""
    valid_dir = DATASET_DIR / "valid" / "axial"
    if valid_dir.exists():
        files = sorted(list(valid_dir.glob("*.npy")))
        return [f.stem for f in files]
    return ["1130", "1131", "1132", "1133", "1134", "1135"]

def load_dataset_exam_planes(exam_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads axial, coronal, and sagittal .npy volumes for a specific exam ID."""
    valid_dir = DATASET_DIR / "valid"
    axial_path = valid_dir / "axial" / f"{exam_id}.npy"
    coronal_path = valid_dir / "coronal" / f"{exam_id}.npy"
    sagittal_path = valid_dir / "sagittal" / f"{exam_id}.npy"
    
    axial_vol = np.load(axial_path) if axial_path.exists() else None
    coronal_vol = np.load(coronal_path) if coronal_path.exists() else None
    sagittal_vol = np.load(sagittal_path) if sagittal_path.exists() else None
    
    return axial_vol, coronal_vol, sagittal_vol

def get_exam_ground_truth(exam_id: str) -> Dict[str, str]:
    """Retrieves ground truth labels for an exam ID across all 3 tasks."""
    labels = {}
    for task in ["acl", "meniscus", "abnormal"]:
        csv_path = DATASET_DIR / f"valid-{task}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, header=None, names=["exam_id", "label"])
                match = df[df["exam_id"].astype(str) == str(exam_id)]
                if not match.empty:
                    val = int(match.iloc[0]["label"])
                    labels[task] = "Positive (Injury)" if val == 1 else "Negative (Normal)"
                else:
                    labels[task] = "Unknown"
            except Exception:
                labels[task] = "Unknown"
        else:
            labels[task] = "Unknown"
    return labels
