"""
Hybrid Enhanced Plane Expert Architecture.
Combines Pretrained CNN Backbone (DenseNet121 / AlexNet / ResNet18 / EfficientNet-B0),
Custom 4-block CNN, CBAM (Channel and Spatial Attention), and Attention-based Slice Aggregation.
Ported directly from research notebook.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    alexnet, AlexNet_Weights,
    densenet121, DenseNet121_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)


class ChannelAttention(nn.Module):
    """Channel Attention Module: Learns WHICH feature channels are important."""
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
    """Spatial Attention Module: Learns WHERE in the MRI slice to focus."""
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
    """Convolutional Block Attention Module combining channel and spatial attention."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CustomCNN(nn.Module):
    """Custom 4-block CNN for MRI-specific representation extraction."""
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
    """Attention-based multi-slice sequence aggregation."""
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
    """Factory for feature extraction backbones."""
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
    Proposed Hybrid Multiplane Feature Expert.
    Fuses deep pretrained backbone activations with custom MRI conv features,
    filtered through CBAM attention and aggregated across depth slices with attention.
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

        if self.config.get("use_cbam", True) and hasattr(self, "cbam"):
            backbone_feat = self.cbam(backbone_feat)

        backbone_feat = self.backbone_pool(backbone_feat).flatten(1)

        if self.config.get("use_custom_cnn", True) and hasattr(self, "custom_cnn"):
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

        if self.config.get("use_slice_attention", True) and hasattr(self, "slice_attention"):
            aggregated = []
            for i in range(b):
                agg, _ = self.slice_attention(slice_features[i])
                aggregated.append(agg)
            v = torch.stack(aggregated, dim=0)
        else:
            v = torch.amax(slice_features, dim=1)

        logit = self.head(v).squeeze(1)
        return logit
