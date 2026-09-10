"""
Preprocessing Pipeline for TriPlane MRI Volumes.
Ported directly from verified training notebooks:
1. Shape & contract validation (S, H, W).
2. Intensity normalization (min-max -> z-score).
3. Hybrid depth standardization (target_slices=25, center preservation + entropy selection).
4. Per-slice bilinear resize (224x224), 3-channel expansion, tensor intensity & ImageNet normalization.
"""

import io
import base64
from typing import Tuple, List, Optional
import numpy as np
from scipy.stats import entropy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class PreprocessingError(Exception):
    """Raised when an input MRI volume fails contract or preprocessing validation."""
    pass


# ============================================================================
# 1. INPUT CONTRACT VALIDATION
# ============================================================================

def validate_mri_volume(volume: np.ndarray, plane_name: str = "volume") -> np.ndarray:
    """
    Validates that input array matches required MRI volume contract:
    - Must be a 3D numpy array (S, H, W)
    - Must have at least 1 slice (S >= 1)
    - Spatial dimensions must be positive (H > 0, W > 0)
    """
    if not isinstance(volume, np.ndarray):
        raise PreprocessingError(f"Input for {plane_name} is not a valid numpy array (got {type(volume).__name__})")

    if volume.ndim != 3:
        raise PreprocessingError(
            f"Input volume for {plane_name} must have shape (S, H, W) with ndim=3, got shape {volume.shape} (ndim={volume.ndim})"
        )

    s, h, w = volume.shape
    if s < 1 or h < 1 or w < 1:
        raise PreprocessingError(
            f"Invalid dimensions for {plane_name}: shape is {volume.shape}. All dimensions must be >= 1."
        )

    if not np.issubdtype(volume.dtype, np.number):
        raise PreprocessingError(f"Input for {plane_name} must have numeric dtype, got {volume.dtype}")

    return volume


# ============================================================================
# 2. INTENSITY NORMALIZATION (VOLUME LEVEL)
# ============================================================================

def min_max_normalize(volume: np.ndarray) -> np.ndarray:
    """Min-max normalization to scale pixel values to [0, 1]."""
    v_min = float(volume.min())
    v_max = float(volume.max())
    if v_max - v_min > 0:
        return (volume - v_min) / (v_max - v_min)
    return volume


def z_score_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization to standardize pixel intensity distributions."""
    mean = float(volume.mean())
    std = float(volume.std())
    if std > 0:
        return (volume - mean) / std
    return volume - mean


def intensity_normalize(volume: np.ndarray) -> np.ndarray:
    """Combined volume-level intensity normalization: min-max -> z-score."""
    volume = min_max_normalize(volume)
    volume = z_score_normalize(volume)
    return volume


# ============================================================================
# 3. VOLUME STANDARDIZATION & HYBRID SLICE SELECTION
# ============================================================================

def calculate_slice_entropy(slice_2d: np.ndarray) -> float:
    """Calculate Shannon entropy of a 2D slice. Higher entropy = richer anatomical content."""
    s_min = float(slice_2d.min())
    s_max = float(slice_2d.max())
    denom = s_max - s_min + 1e-8
    slice_norm = ((slice_2d - s_min) / denom * 255).astype(np.uint8)
    hist, _ = np.histogram(slice_norm.flatten(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(entropy(hist))


def hybrid_slice_selection(volume: np.ndarray, target_slices: int = 25, center_ratio: float = 0.4) -> np.ndarray:
    """
    Hybrid slice selection combining center preservation + entropy-based selection:
    1. ALWAYS preserves center slices (critical pathology regions for ACL/meniscus)
    2. Fills remaining slots with high-entropy slices from periphery
    3. Maintains anatomical continuity by sorting final indices
    """
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
        best_idx = remaining[int(np.argmax(entropies_remaining))]
        all_indices.append(best_idx)
        all_indices = sorted(all_indices)

    return volume[all_indices[:target_slices]]


def standardize_volume(volume: np.ndarray, target_slices: int = 25, center_ratio: float = 0.4) -> np.ndarray:
    """Standardizes volume depth to fixed target_slices (zero-padding or hybrid selection)."""
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
        return hybrid_slice_selection(volume, target_slices=target_slices, center_ratio=center_ratio)


# ============================================================================
# 4. SLICE & TENSOR TRANSFORMS
# ============================================================================

def min_max_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    v_min = x.min()
    v_max = x.max()
    if v_max - v_min > 0:
        return (x - v_min) / (v_max - v_min)
    return x


def z_score_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean()
    std = x.std()
    if std > 0:
        return (x - mean) / std
    return x - mean


IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def preprocess_slice_to_tensor(slice_2d: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """
    Transforms a single 2D slice into a normalized 3-channel PyTorch tensor (3, H, W).
    Follows notebook's NPYSliceTransform (valid_tf) path.
    """
    x = torch.from_numpy(slice_2d).float().unsqueeze(0)  # (1, H, W)
    x = F.interpolate(x.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = x.repeat(3, 1, 1)  # (3, H, W)

    x = min_max_normalize_tensor(x)
    x = z_score_normalize_tensor(x)
    x = IMAGENET_NORMALIZE(x)
    return x


def preprocess_volume_to_tensor(
    volume: np.ndarray,
    target_slices: int = 25,
    img_size: int = 224,
    plane_name: str = "volume"
) -> torch.Tensor:
    """
    Full end-to-end preprocessing for one plane:
    Validates -> Intensity normalizes -> Standardizes depth -> Transforms slices -> Batches (1, K, 3, H, W).
    """
    volume = validate_mri_volume(volume, plane_name=plane_name)
    volume_norm = intensity_normalize(volume)
    volume_std = standardize_volume(volume_norm, target_slices=target_slices)

    slices = [preprocess_slice_to_tensor(volume_std[s], img_size=img_size) for s in range(volume_std.shape[0])]
    volume_tensor = torch.stack(slices, dim=0)  # (K, 3, H, W)
    return volume_tensor.unsqueeze(0)  # (1, K, 3, H, W)


# ============================================================================
# 5. SERIALIZATION / DESERIALIZATION HELPERS
# ============================================================================

def parse_npy_bytes(data: bytes, plane_name: str = "volume") -> np.ndarray:
    """Loads a numpy array from binary bytes buffer."""
    try:
        arr = np.load(io.BytesIO(data))
        return validate_mri_volume(arr, plane_name=plane_name)
    except PreprocessingError:
        raise
    except Exception as e:
        raise PreprocessingError(f"Failed to parse .npy binary data for {plane_name}: {str(e)}")


def parse_npy_base64(b64_str: str, plane_name: str = "volume") -> np.ndarray:
    """Loads a numpy array from base64 encoded string."""
    try:
        # Strip potential data URL prefix
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        decoded = base64.b64decode(b64_str)
        return parse_npy_bytes(decoded, plane_name=plane_name)
    except PreprocessingError:
        raise
    except Exception as e:
        raise PreprocessingError(f"Invalid base64 payload for {plane_name}: {str(e)}")
