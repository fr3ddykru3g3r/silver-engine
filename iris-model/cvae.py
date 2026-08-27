from __future__ import annotations

"""Compact conditional VAE for positive LOS magnetograms.

The diffusion pilot showed that unconstrained from-noise synthesis can match scalar
flux summaries before it matches the full image distribution.  This model therefore
starts from a reconstruction objective: the decoder must first reproduce real HMI
magnetograms, and only then is the latent distribution used for synthesis.

Latitude is an explicit conditioning variable because Hale/Joy geometry is
hemisphere-dependent.  No flare outcome beyond the already-frozen positive training
selection is supplied to the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = 8
    while ch % g and g > 1:
        g //= 2
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n1 = _gn(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = ResBlock(cin, cout)
        self.down = nn.Conv2d(cout, cout, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.block(x))


class Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)
        self.block = ResBlock(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.up(x))


class MagnetogramCVAE(nn.Module):
    def __init__(self, base: int = 16, latent_dim: int = 64):
        super().__init__()
        self.base = int(base)
        self.latent_dim = int(latent_dim)
        ch = [base, base * 2, base * 4, base * 6, base * 8]

        self.stem = nn.Conv2d(1, ch[0], 3, padding=1)
        self.d1 = Down(ch[0], ch[1])   # 128 -> 64
        self.d2 = Down(ch[1], ch[2])   # 64 -> 32
        self.d3 = Down(ch[2], ch[3])   # 32 -> 16
        self.d4 = Down(ch[3], ch[4])   # 16 -> 8
        self.d5 = Down(ch[4], ch[4])   # 8 -> 4
        flat = ch[4] * 4 * 4
        self.mu = nn.Linear(flat, latent_dim)
        self.logvar = nn.Linear(flat, latent_dim)

        self.lat_embed = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, latent_dim)
        )
        self.from_z = nn.Linear(latent_dim, flat)
        self.u1 = Up(ch[4], ch[4])     # 4 -> 8
        self.u2 = Up(ch[4], ch[3])     # 8 -> 16
        self.u3 = Up(ch[3], ch[2])     # 16 -> 32
        self.u4 = Up(ch[2], ch[1])     # 32 -> 64
        self.u5 = Up(ch[1], ch[0])     # 64 -> 128
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)

    @staticmethod
    def latitude_condition(latitude_deg: torch.Tensor) -> torch.Tensor:
        # HMI active regions are concentrated well within +/-45 deg.  Clamping only
        # bounds the neural conditioning variable; the stored physical latitude is
        # never modified.
        return torch.clamp(latitude_deg.float() / 45.0, -1.25, 1.25).view(-1, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.d1(h); h = self.d2(h); h = self.d3(h); h = self.d4(h); h = self.d5(h)
        h = h.flatten(1)
        return self.mu(h), torch.clamp(self.logvar(h), -8.0, 5.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            eps = torch.randn_like(mu)
        else:
            eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor, latitude_deg: torch.Tensor) -> torch.Tensor:
        lat = self.lat_embed(self.latitude_condition(latitude_deg))
        h = self.from_z(z + lat).view(len(z), self.base * 8, 4, 4)
        h = self.u1(h); h = self.u2(h); h = self.u3(h); h = self.u4(h); h = self.u5(h)
        return torch.tanh(self.out(h))

    def forward(self, x: torch.Tensor, latitude_deg: torch.Tensor,
                generator: torch.Generator | None = None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        recon = self.decode(z, latitude_deg)
        return recon, mu, logvar


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(mu.square() + logvar.exp() - 1.0 - logvar, dim=1))


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
