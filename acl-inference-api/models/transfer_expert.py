"""
Transfer Learning Only Plane Expert Architecture.
Extracts features using standard ImageNet backbones (DenseNet121 default) with slice max pooling.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .hybrid_expert import get_backbone
except (ImportError, ValueError):
    from models.hybrid_expert import get_backbone


class TransferLearningPlaneExpert(nn.Module):
    """Transfer Learning Plane Expert (Pretrained backbone only, depth max pooling)."""
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
