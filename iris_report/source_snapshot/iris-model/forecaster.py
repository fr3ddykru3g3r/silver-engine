from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        g1 = min(8, cout)
        while cout % g1: g1 -= 1
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(g1, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g1, cout),
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class FlareCNN(nn.Module):
    """Deliberately small downstream forecaster for physics-ablation experiments.

    The model is intentionally not a foundation model: the scientific question is
    whether different synthetic-data constraints transfer useful information under
    matched exposure, not whether scale alone improves the forecast.
    """
    def __init__(self, width: int = 32, dropout: float = 0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, width, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.body = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, 2),
            ResidualBlock(width * 2, width * 2),
            ResidualBlock(width * 2, width * 4, 2),
            ResidualBlock(width * 4, width * 4),
            ResidualBlock(width * 4, width * 8, 2),
            ResidualBlock(width * 8, width * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, 1),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
