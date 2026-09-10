"""
Pydantic Request and Response Schemas for TriPlane Health API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ============================================================================
# 1. SYSTEM & MODEL INFO SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    status: str = Field(...)
    models_loaded: List[str] = Field(...)
    device: str = Field(...)
    total_models_available: int = Field(...)


class PlaneMetricInfo(BaseModel):
    best_auc: float
    loss_type: str
    checkpoint: str


class ModelDetail(BaseModel):
    model_key: str
    name: str
    description: str
    has_attention: bool
    paper_auc: float
    is_loaded: bool
    planes: Dict[str, PlaneMetricInfo]


class ModelsListResponse(BaseModel):
    models: List[ModelDetail]


# ============================================================================
# 2. PREDICTION SCHEMAS
# ============================================================================

class PlaneLogits(BaseModel):
    axial: float = Field(..., description="Axial plane forward pass logit")
    coronal: float = Field(..., description="Coronal plane forward pass logit")
    sagittal: float = Field(..., description="Sagittal plane forward pass logit")


class FusionWeights(BaseModel):
    axial: float = Field(..., description="Logistic regression weight for axial plane")
    coronal: float = Field(..., description="Logistic regression weight for coronal plane")
    sagittal: float = Field(..., description="Logistic regression weight for sagittal plane")
    bias: float = Field(..., description="Logistic regression intercept / bias term")


class PredictRequestJSON(BaseModel):
    model: str = Field("proposed", description="Key in model registry ('proposed', 'transfer_learning', 'custom_cnn')")
    axial: str = Field(..., description="Base64-encoded .npy file bytes or data URL for axial plane")
    coronal: str = Field(..., description="Base64-encoded .npy file bytes or data URL for coronal plane")
    sagittal: str = Field(..., description="Base64-encoded .npy file bytes or data URL for sagittal plane")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for tear classification")
    exam_id: Optional[str] = Field(None, description="Optional exam / patient identifier for audit logging")


class PredictResponse(BaseModel):
    model: str = Field(...)
    probability: float = Field(...)
    predicted_label: str = Field(...)
    threshold: float = Field(0.5)
    plane_logits: PlaneLogits
    fusion_weights: FusionWeights
    has_attention: bool = Field(...)
    exam_id: Optional[str] = None
    execution_time_ms: Optional[float] = None


class CompareResponse(BaseModel):
    results: List[PredictResponse]
    exam_id: Optional[str] = None
    execution_time_ms: Optional[float] = None


# ============================================================================
# 3. GRAD-CAM SCHEMAS
# ============================================================================

class GradCAMRequestJSON(BaseModel):
    model: str = Field("proposed", description="Model to visualize ('proposed' or 'transfer_learning')")
    plane: str = Field("axial", description="MRI plane ('axial', 'coronal', 'sagittal')")
    slice_idx: Optional[int] = Field(None, ge=0, le=24, description="Target slice index (0-24). Defaults to center slice.")
    target_layer: str = Field("cbam", description="Target module ('cbam' or 'backbone')")
    # Provide the plane volume either as plane-specific base64 or all 3
    volume: Optional[str] = Field(None, description="Base64-encoded .npy file for the target plane")
    axial: Optional[str] = None
    coronal: Optional[str] = None
    sagittal: Optional[str] = None
    exam_id: Optional[str] = None


class HotspotCoord(BaseModel):
    x: int
    y: int


class GradCAMResponse(BaseModel):
    model: str
    plane: str
    slice_idx: int
    total_slices: int
    confidence: float
    target_layer: str
    hotspot: HotspotCoord
    original_image: str = Field(..., description="Base64 PNG data URL of grayscale MRI slice")
    heatmap_image: str = Field(..., description="Base64 PNG data URL of JET colormap Grad-CAM")
    overlay_image: str = Field(..., description="Base64 PNG data URL of fused overlay")


# ============================================================================
# 4. AUDIT TRAIL SCHEMAS
# ============================================================================

class AuditLogItem(BaseModel):
    id: int
    timestamp: str
    exam_id: Optional[str]
    model: str
    probability: float
    predicted_label: str
    plane_logits: Dict[str, float]
    execution_time_ms: float


class AuditLogsResponse(BaseModel):
    total_records: int
    logs: List[AuditLogItem]
