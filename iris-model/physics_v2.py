from __future__ import annotations

"""V2 differentiable magnetic-structure losses.

Design goals:
1. Keep quiet/background pixels from dominating polarity centroids or PIL masks.
2. Treat strong-PIL gradient as a distribution, not one scalar mean.
3. Express spatial radii in physical Mm after fixed-FOV resampling.
4. Keep the hard scientific diagnostic independent from this smooth training proxy.
"""

import torch
import torch.nn.functional as F

EPS = 1e-6


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.mean((x.mean(0) - y.mean(0)) ** 2)
    xy = torch.cdist(x, y, p=2).mean()
    xx = torch.cdist(x, x, p=2).mean()
    yy = torch.cdist(y, y, p=2).mean()
    return 2 * xy - xx - yy


def quantile_distribution_loss(x: torch.Tensor, y: torch.Tensor,
                               quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)) -> torch.Tensor:
    """Match several marginal quantiles of a descriptor vector."""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    q = torch.as_tensor(quantiles, device=x.device, dtype=x.dtype)
    qx = torch.quantile(x, q, dim=0)
    qy = torch.quantile(y.detach(), q, dim=0)
    return F.smooth_l1_loss(qx, qy)


def distribution_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Blend global distribution matching with tail/quantile matching.

    Energy distance supplies gradients across the whole batch; quantiles force the
    generator to reproduce tails rather than merely the centre of the distribution.
    """
    return 0.5 * energy_distance(x, y.detach()) + 0.5 * quantile_distribution_loss(x, y)


def _strong_polarity_weights(b: torch.Tensor, threshold_g: float = 150.0,
                             temperature_g: float = 50.0):
    if b.ndim == 3:
        b = b[:, None]
    # ReLU makes exact quiet-Sun/background field contribute zero mass. The smooth
    # threshold gate controls how rapidly pixels become relevant around 150 G.
    p_gate = torch.sigmoid((b - threshold_g) / temperature_g)
    n_gate = torch.sigmoid((-b - threshold_g) / temperature_g)
    pos = p_gate * F.relu(b)
    neg = n_gate * F.relu(-b)
    return pos, neg


def polarity_geometry_descriptor_v2(b: torch.Tensor, latitude_deg: torch.Tensor,
                                    threshold_g: float = 150.0,
                                    temperature_g: float = 50.0) -> torch.Tensor:
    """Strong-field bipole orientation/separation descriptor."""
    if b.ndim == 3:
        b = b[:, None]
    _, _, h, w = b.shape
    yy = torch.linspace(-1, 1, h, device=b.device, dtype=b.dtype).view(1, 1, h, 1)
    xx = torch.linspace(-1, 1, w, device=b.device, dtype=b.dtype).view(1, 1, 1, w)
    pos, neg = _strong_polarity_weights(b, threshold_g, temperature_g)
    pnorm = pos.sum((2, 3)) + EPS
    nnorm = neg.sum((2, 3)) + EPS
    px = (pos * xx).sum((2, 3)) / pnorm
    py = (pos * yy).sum((2, 3)) / pnorm
    nx = (neg * xx).sum((2, 3)) / nnorm
    ny = (neg * yy).sum((2, 3)) / nnorm
    dx = (px - nx).squeeze(1)
    dy = (py - ny).squeeze(1)
    sep = torch.sqrt(dx.square() + dy.square() + EPS)
    ux = dx / sep
    uy = dy / sep
    hemi = torch.where(latitude_deg >= 0, torch.ones_like(latitude_deg), -torch.ones_like(latitude_deg)).to(b.dtype)
    return torch.stack([hemi * ux, hemi * uy, torch.log1p(sep)], dim=1)


def population_distribution_loss_v2(fake_b: torch.Tensor, real_b: torch.Tensor,
                                    latitude_deg: torch.Tensor) -> torch.Tensor:
    losses = []
    for mask in (latitude_deg >= 0, latitude_deg < 0):
        if int(mask.sum()) >= 3:
            f = polarity_geometry_descriptor_v2(fake_b[mask], latitude_deg[mask])
            r = polarity_geometry_descriptor_v2(real_b[mask].detach(), latitude_deg[mask])
            losses.append(distribution_loss(f, r))
    return torch.stack(losses).mean() if losses else fake_b.sum() * 0.0


def _central_gradient(b: torch.Tensor, pixel_mm: float):
    bx = F.pad(b, (1, 1, 0, 0), mode='replicate')
    by = F.pad(b, (0, 0, 1, 1), mode='replicate')
    gx = (bx[:, :, :, 2:] - bx[:, :, :, :-2]) / (2.0 * pixel_mm)
    gy = (by[:, :, 2:, :] - by[:, :, :-2, :]) / (2.0 * pixel_mm)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def _soft_dilate(x: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px <= 0:
        return x
    k = 2 * radius_px + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=radius_px)


def soft_pil_contact(b: torch.Tensor, strong_field_g: float = 150.0,
                     membership_temp_g: float = 40.0,
                     contact_radius_px: int = 2) -> torch.Tensor:
    """Smooth strong-field opposite-polarity proximity mask.

    A soft logical-AND (relu(dp+dn-1)) suppresses the non-zero sigmoid background
    that would otherwise accumulate across tens of thousands of quiet pixels.
    """
    if b.ndim == 3:
        b = b[:, None]
    pos = torch.sigmoid((b - strong_field_g) / membership_temp_g)
    neg = torch.sigmoid((-b - strong_field_g) / membership_temp_g)
    dp = _soft_dilate(pos, contact_radius_px)
    dn = _soft_dilate(neg, contact_radius_px)
    return F.relu(dp + dn - 1.0)


def _weighted_mean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (v * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) + EPS)


def strong_pil_gradient_descriptor_v2(
    b: torch.Tensor,
    pixel_mm: float = 2.0,
    strong_field_g: float = 150.0,
    membership_temp_g: float = 40.0,
    contact_radius_mm: float = 4.0,
    high_gradient_g_per_mm: float = 250.0,
    high_gradient_temp: float = 40.0,
    exceedance_thresholds=(100.0, 250.0, 500.0),
) -> torch.Tensor:
    """Distribution-sensitive strong-PIL-gradient descriptor.

    Every component derives from |grad B| near a strong-field LOS polarity inversion
    line; no vector-field quantity is introduced.
    """
    if b.ndim == 3:
        b = b[:, None]
    radius_px = max(1, int(round(contact_radius_mm / pixel_mm)))
    contact = soft_pil_contact(b, strong_field_g, membership_temp_g, radius_px)
    grad = _central_gradient(b, pixel_mm)

    mean_g = _weighted_mean(grad, contact)
    rms_g = torch.sqrt(_weighted_mean(grad.square(), contact) + EPS)

    tail_gate = torch.sigmoid((grad - high_gradient_g_per_mm) / high_gradient_temp)
    tail_w = contact * tail_gate
    tail_mean = _weighted_mean(grad, tail_w)

    feats = [
        torch.log1p(mean_g / 50.0),
        torch.log1p(rms_g / 50.0),
        torch.log1p(tail_mean / 50.0),
    ]
    denom = contact.sum((1, 2, 3)) + EPS
    for thr in exceedance_thresholds:
        ex = torch.sigmoid((grad - float(thr)) / high_gradient_temp)
        feats.append((contact * ex).sum((1, 2, 3)) / denom)
    return torch.stack(feats, dim=1)


def pil_distribution_loss_v2(fake_b: torch.Tensor, real_b: torch.Tensor,
                             pixel_mm: float = 2.0) -> torch.Tensor:
    f = strong_pil_gradient_descriptor_v2(fake_b, pixel_mm=pixel_mm)
    r = strong_pil_gradient_descriptor_v2(real_b.detach(), pixel_mm=pixel_mm)
    return distribution_loss(f, r)


def physics_components_v2(fake_b: torch.Tensor, real_b: torch.Tensor,
                          latitude_deg: torch.Tensor, pixel_mm: float = 2.0):
    return {
        'hj': population_distribution_loss_v2(fake_b, real_b, latitude_deg),
        'pil': pil_distribution_loss_v2(fake_b, real_b, pixel_mm=pixel_mm),
    }
