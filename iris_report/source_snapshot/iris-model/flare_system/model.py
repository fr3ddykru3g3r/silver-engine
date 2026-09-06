from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def magnetogram_channels(x: torch.Tensor) -> torch.Tensor:
    """Create signed, unsigned, and differentiable PIL-aware channels."""
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected N x 1 x H x W magnetograms, received {tuple(x.shape)}")
    signed = x
    unsigned = x.abs()
    local_signed = F.avg_pool2d(signed, 7, stride=1, padding=3)
    local_unsigned = F.avg_pool2d(unsigned, 7, stride=1, padding=3)
    mixed_polarity = (local_unsigned - local_signed.abs()).clamp_min(0.0)
    dx = F.pad(signed[:, :, :, 1:] - signed[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(signed[:, :, 1:, :] - signed[:, :, :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1e-8)
    pil = torch.tanh(4.0 * mixed_polarity * gradient)
    return torch.cat([signed, unsigned, pil], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(x) + self.skip(x))


class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.local = ResidualBlock(channels, half, dilation=1)
        self.context = ResidualBlock(channels, channels - half, dilation=2)
        self.mix = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.mix(torch.cat([self.local(x), self.context(x)], dim=1)) + x)


class HybridFlareNet(nn.Module):
    """Multi-scale magnetogram encoder fused with four SHARP/geometry features."""

    def __init__(self, width: int = 32, feature_dim: int = 4, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(width, width),
            MultiScaleBlock(width),
            ResidualBlock(width, width * 2, stride=2),
            MultiScaleBlock(width * 2),
            ResidualBlock(width * 2, width * 4, stride=2),
            MultiScaleBlock(width * 4),
            ResidualBlock(width * 4, width * 8, stride=2),
            MultiScaleBlock(width * 8),
        )
        self.attention = nn.Sequential(nn.Conv2d(width * 8, 1, 1), nn.Sigmoid())
        image_dim = width * 16
        self.physics = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, 1)
        self.auxiliary_physics = nn.Linear(image_dim, 2)

    def forward(self, magnetogram: torch.Tensor, physics: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.stem(magnetogram_channels(magnetogram)))
        encoded = encoded * (1.0 + self.attention(encoded))
        average = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(encoded, 1).flatten(1)
        image_features = torch.cat([average, maximum], dim=1)
        fused = self.fusion(torch.cat([image_features, self.physics(physics)], dim=1))
        return {
            "logit": self.classifier(fused).squeeze(-1),
            "aux_physics": self.auxiliary_physics(image_features),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
