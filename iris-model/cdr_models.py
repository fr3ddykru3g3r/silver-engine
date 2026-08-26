from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Six-convolution magnetogram encoder inspired by Wu et al. (2026).

    The paper specifies one 11x11 / subsequent 3x3 convolutional design with
    batch normalization, ReLU and four max-pooling stages, but not every channel
    width in the article text. Channel widths below are therefore an explicit
    adaptation to 128x128 one-channel SHARP inputs, not a claim of bit-for-bit
    reproduction of their unpublished implementation.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        chans = [1, 32, 64, 96, 128, 192, 256]
        blocks = []
        for i in range(6):
            k = 11 if i == 0 else 3
            p = k // 2
            blocks += [nn.Conv2d(chans[i], chans[i+1], k, padding=p, bias=False),
                       nn.BatchNorm2d(chans[i+1]), nn.ReLU(inplace=True)]
            if i < 4:
                blocks += [nn.MaxPool2d(2)]
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(chans[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.net(x)).flatten(1)
        return self.proj(z)


class CDRImageCNN(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = ConvEncoder(hidden)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


class CDRImageBiLSTM(nn.Module):
    """Shared six-layer CNN frame encoder followed by bidirectional LSTM."""
    def __init__(self, frame_dim: int = 192, lstm_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = ConvEncoder(frame_dim)
        self.rnn = nn.LSTM(frame_dim, lstm_hidden, num_layers=1, batch_first=True,
                           bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(2*lstm_hidden, 128),
                                  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,1,H,W
        b,t,c,h,w = x.shape
        z = self.encoder(x.reshape(b*t,c,h,w)).reshape(b,t,-1)
        z,_ = self.rnn(z)
        # Combine both temporal directions at final sequence position.
        return self.head(z[:, -1]).squeeze(-1)


class FeatureTransformer(nn.Module):
    """Temporal Transformer for 2- or 10-feature SHARP sequences.

    The paper uses four Transformer encoder modules and treats the feature vector
    at each time step as a token along the temporal dimension. We preserve that
    structure and use a compact PyTorch implementation suitable for our locked
    hourly 24-step sequences.
    """
    def __init__(self, n_features: int, seq_len: int = 24, d_model: int = 64,
                 nhead: int = 4, layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.seq_len = seq_len
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=d_model*2,
                                         dropout=dropout, batch_first=True,
                                         norm_first=False, activation='relu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(seq_len*d_model, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(96, 1))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(f'expected sequence length {self.seq_len}, got {x.shape[1]}')
        z = self.inp(x) + self.pos
        z = self.encoder(z)
        return self.head(z).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
