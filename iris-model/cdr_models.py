from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """Six-convolution magnetogram encoder inspired by Wu et al. (2026).

    The article specifies six convolutions (first 11x11, subsequent 3x3), BN,
    ReLU, and four 2x2 max-pooling layers. It does not publish all filter counts
    in the article text, so widths below are an explicitly documented adaptation
    for our 128x128 one-channel SHARP preprocessing rather than a claim of a
    bit-for-bit reproduction of unavailable implementation details.
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
        return self.head(z[:, -1]).squeeze(-1)


class TemporalBatchNorm(nn.Module):
    """Batch-normalize embedding channels while preserving B,T,D layout."""
    def __init__(self, d_model: int):
        super().__init__(); self.bn=nn.BatchNorm1d(d_model)
    def forward(self,x):
        return self.bn(x.transpose(1,2)).transpose(1,2)


class PaperTransformerBlock(nn.Module):
    """MHA + residual + BN + two-layer MLP + residual.

    This mirrors the structure described in Fig. 2 / Section 3.2 of Wu et al.
    The article does not expose every hidden dimension/head count, so d_model,
    nhead and MLP width remain reproducible project-level choices.
    """
    def __init__(self,d_model:int,nhead:int,dropout:float):
        super().__init__()
        self.attn=nn.MultiheadAttention(d_model,nhead,dropout=dropout,batch_first=True)
        self.bn=TemporalBatchNorm(d_model)
        self.mlp=nn.Sequential(nn.Linear(d_model,d_model*2),nn.ReLU(),nn.Dropout(dropout),
                               nn.Linear(d_model*2,d_model),nn.Dropout(dropout))
    def forward(self,x):
        a,_=self.attn(x,x,x,need_weights=False)
        z=x+a
        n=self.bn(z)
        return z+self.mlp(n)


class FeatureTransformer(nn.Module):
    """Four-encoder temporal Transformer for 2- or 10-feature SHARP states.

    Wu et al. use 40 observations collected every 36 minutes. Our default now
    follows that temporal structure exactly. A learned linear projection and
    positional embedding implement the paper's Patches/Patch_encoder role.
    """
    def __init__(self, n_features: int, seq_len: int = 40, d_model: int = 64,
                 nhead: int = 4, layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.seq_len=seq_len
        self.inp=nn.Linear(n_features,d_model)
        self.pos=nn.Parameter(torch.zeros(1,seq_len,d_model))
        self.blocks=nn.ModuleList([PaperTransformerBlock(d_model,nhead,dropout) for _ in range(layers)])
        # Paper NN module: Flatten + 3 Dense + 3 BN + 3 Dropout + Softmax.
        # We output one binary logit; sigmoid is applied externally.
        self.fc1=nn.Linear(seq_len*d_model,256);self.bn1=nn.BatchNorm1d(256);self.do1=nn.Dropout(dropout)
        self.fc2=nn.Linear(256,96);self.bn2=nn.BatchNorm1d(96);self.do2=nn.Dropout(dropout)
        self.fc3=nn.Linear(96,32);self.bn3=nn.BatchNorm1d(32);self.do3=nn.Dropout(dropout)
        self.out=nn.Linear(32,1);self.act=nn.ReLU()
        nn.init.normal_(self.pos,std=0.02)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.shape[1]!=self.seq_len:
            raise ValueError(f'expected sequence length {self.seq_len}, got {x.shape[1]}')
        z=self.inp(x)+self.pos
        for block in self.blocks:z=block(z)
        z=z.flatten(1)
        z=self.do1(self.act(self.bn1(self.fc1(z))))
        z=self.do2(self.act(self.bn2(self.fc2(z))))
        z=self.do3(self.act(self.bn3(self.fc3(z))))
        return self.out(z).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
