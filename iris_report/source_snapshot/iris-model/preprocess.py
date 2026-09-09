from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits


@dataclass(frozen=True)
class PreprocessConfig:
    output_size: int = 128
    fov_mm: float = 256.0
    clip_gauss: float = 3000.0
    asinh_scale_gauss: float = 250.0
    centroid_threshold_gauss: float = 100.0


def _first_2d_array(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ValueError(f"No 2-D FITS image found in {path}")


def native_scale_mm(cdelt_deg: float, rsun_ref_m: float) -> float:
    """Approximate CEA pixel scale along the solar surface in Mm/pixel."""
    x = abs(float(cdelt_deg)) * math.pi / 180.0 * float(rsun_ref_m) / 1e6
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"Invalid physical pixel scale: CDELT={cdelt_deg}, RSUN_REF={rsun_ref_m}")
    return x


def _flux_centroid(b: np.ndarray, threshold: float) -> tuple[float, float]:
    w = np.abs(np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float64)
    w[w < threshold] = 0.0
    if w.sum() <= 0:
        return (0.5 * (b.shape[0] - 1), 0.5 * (b.shape[1] - 1))
    yy, xx = np.indices(b.shape)
    return float((yy * w).sum() / w.sum()), float((xx * w).sum() / w.sum())


def _crop_or_pad(arr: np.ndarray, cy: float, cx: float, h: int, w: int) -> np.ndarray:
    h = max(2, int(h)); w = max(2, int(w))
    y0 = int(round(cy - (h - 1) / 2)); x0 = int(round(cx - (w - 1) / 2))
    y1 = y0 + h; x1 = x0 + w
    out = np.zeros((h, w), dtype=np.float32)
    sy0, sy1 = max(0, y0), min(arr.shape[0], y1)
    sx0, sx1 = max(0, x0), min(arr.shape[1], x1)
    if sy1 > sy0 and sx1 > sx0:
        oy0, ox0 = sy0 - y0, sx0 - x0
        out[oy0:oy0 + (sy1 - sy0), ox0:ox0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out


def standardize_physical_fov(
    b_gauss: np.ndarray,
    cdelt1_deg: float,
    cdelt2_deg: float,
    rsun_ref_m: float,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """Map a variable-size SHARP patch to a fixed physical FOV.

    The crop is centered on the unsigned-flux centroid. The final grid always spans
    cfg.fov_mm x cfg.fov_mm, so the generated/model pixel scale is exactly
    cfg.fov_mm / cfg.output_size Mm/pixel. This is essential for a PIL-gradient loss.
    """
    b = np.nan_to_num(np.asarray(b_gauss, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sy = native_scale_mm(cdelt2_deg, rsun_ref_m)
    sx = native_scale_mm(cdelt1_deg, rsun_ref_m)
    crop_h = int(round(cfg.fov_mm / sy))
    crop_w = int(round(cfg.fov_mm / sx))
    cy, cx = _flux_centroid(b, cfg.centroid_threshold_gauss)
    crop = _crop_or_pad(b, cy, cx, crop_h, crop_w)
    t = torch.from_numpy(crop)[None, None]
    t = F.interpolate(t, size=(cfg.output_size, cfg.output_size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float32)


def normalize_gauss(b: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    b = np.clip(np.asarray(b, dtype=np.float32), -cfg.clip_gauss, cfg.clip_gauss)
    denom = np.arcsinh(cfg.clip_gauss / cfg.asinh_scale_gauss)
    return (np.arcsinh(b / cfg.asinh_scale_gauss) / denom).astype(np.float32)


def denormalize_gauss(x: torch.Tensor, cfg: PreprocessConfig = PreprocessConfig()) -> torch.Tensor:
    denom = math.asinh(cfg.clip_gauss / cfg.asinh_scale_gauss)
    return cfg.asinh_scale_gauss * torch.sinh(torch.clamp(x, -1.0, 1.0) * denom)


def preprocess_fits(
    path: str | Path,
    cdelt1_deg: float,
    cdelt2_deg: float,
    rsun_ref_m: float,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = _first_2d_array(path)
    fixed = standardize_physical_fov(raw, cdelt1_deg, cdelt2_deg, rsun_ref_m, cfg)
    norm = normalize_gauss(fixed, cfg)
    return torch.from_numpy(norm)[None], torch.from_numpy(fixed)[None]
