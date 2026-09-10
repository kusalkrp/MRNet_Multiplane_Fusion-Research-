"""
Custom CNN Only Plane Expert Architecture.
Uses a 4-block convolutional network trained from scratch with slice max pooling.
"""

from typing import Dict
import torch
import torch.nn as nn
try:
    from .hybrid_expert import CustomCNN
except (ImportError, ValueError):
    from models.hybrid_expert import CustomCNN


class CustomCNNPlaneExpert(nn.Module):
    """Custom CNN Plane Expert (Trained from scratch, depth max pooling)."""
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
