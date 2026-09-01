#!/usr/bin/env python3
"""Continue the interrupted BASE generator run from a numbered checkpoint.

This helper keeps the notebook's BASE hyperparameters, uses only the verified
local FITS cache, and limits PyTorch CPU threads so a laptop-safe continuation
does not monopolize the machine.  The original checkpoint does not contain
optimizer/RNG state, so the resulting continuation is recorded as a resumed
run rather than silently presented as a bit-for-bit continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--fits-source", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--prepared-cache", required=True)
    args = ap.parse_args()

    source_dir = Path(args.source_dir).resolve()
    model_dir = source_dir / "iris-model"
    sys.path.insert(0, str(model_dir))

    from data import build_records, cache_records
    from generator import ConditionalUNet, Diffusion
    from preprocess import denormalize_gauss, preprocess_fits
    from physics_v2 import population_distribution_loss_v2, pil_distribution_loss_v2
    from generic_fidelity import generic_distribution_loss
    from train_generator_v2 import (
        collate,
        group_temporal_subset,
        make_rng,
        mixed_sampler,
        next_or_restart,
        positive_group_sampler,
        randn_like_with,
        ema_update,
    )

    if args.target_steps <= 0:
        raise ValueError("target-steps must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    start_step = int(checkpoint.get("steps", 0))
    if start_step <= 0:
        raise RuntimeError(f"Checkpoint has no positive step count: {checkpoint_path}")
    if start_step >= args.target_steps:
        raise RuntimeError(
            f"Checkpoint is already at step {start_step}; target is {args.target_steps}"
        )

    seed = int(checkpoint.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["IRIS_FITS_SOURCE"] = str(Path(args.fits_source).resolve())
    os.environ["IRIS_REQUIRE_LOCAL_FITS"] = "1"

    evidence_dir = Path(args.evidence_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out = out_root / "base"
    out.mkdir(parents=True, exist_ok=True)

    full = build_records(evidence_dir, "train")
    selected = group_temporal_subset(
        full,
        int(checkpoint.get("per_group", 4)),
        int(checkpoint.get("positive_slots", 4)),
        seed,
    )
    positives = selected[selected.label_m1plus_24h.eq(1)].copy().reset_index(drop=True)
    if positives.region_group_id.nunique() < 3:
        raise RuntimeError("Not enough positive physical regions for BASE training")

    selected = cache_records(selected, out_root / "cache/generator", workers=2)
    positives = selected[selected.label_m1plus_24h.eq(1)].copy().reset_index(drop=True)

    class PreparedDataset(Dataset):
        def __init__(self, records: pd.DataFrame, x_array: np.ndarray,
                     raw_array: np.ndarray, indices: list[int] | None = None):
            self.records = records.reset_index(drop=True)
            self.x_array = x_array
            self.raw_array = raw_array
            self.indices = list(range(len(self.records))) if indices is None else indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            row_index = self.indices[i]
            row = self.records.iloc[row_index]
            return {
                "x": torch.from_numpy(np.array(self.x_array[row_index], copy=True)).float(),
                "raw_gauss": torch.from_numpy(np.array(self.raw_array[row_index], copy=True)).float(),
                "y": torch.tensor(float(row.label_m1plus_24h), dtype=torch.float32),
                "latitude": torch.tensor(float(row.latitude_deg), dtype=torch.float32),
                "group": str(row.region_group_id),
                "sample_id": str(row.sample_id),
            }

    def prepare_cache(records: pd.DataFrame, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = cache_dir / "metadata.json"
        x_path = cache_dir / "x.npy"
        raw_path = cache_dir / "raw.npy"
        sample_ids = [str(x) for x in records.sample_id]
        metadata = {
            "sample_ids": sample_ids,
            "shape": [len(records), 1, 128, 128],
            "dtype": "float32",
        }
        if metadata_path.is_file() and x_path.is_file() and raw_path.is_file():
            try:
                saved = json.loads(metadata_path.read_text())
                x_existing = np.load(x_path, mmap_mode="r")
                raw_existing = np.load(raw_path, mmap_mode="r")
                if saved == metadata and tuple(x_existing.shape) == tuple(metadata["shape"]):
                    print(f"using prepared tensor cache: {cache_dir}", flush=True)
                    return x_existing, raw_existing
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        print(f"preparing fixed tensor cache: {len(records)} FITS files", flush=True)
        x_part = cache_dir / "x.npy.part"
        raw_part = cache_dir / "raw.npy.part"
        x_mem = np.lib.format.open_memmap(
            x_part, mode="w+", dtype=np.float32, shape=tuple(metadata["shape"])
        )
        raw_mem = np.lib.format.open_memmap(
            raw_part, mode="w+", dtype=np.float32, shape=tuple(metadata["shape"])
        )
        for i, row in enumerate(records.itertuples(index=False), 1):
            norm, raw = preprocess_fits(
                row.fits_path, float(row.CDELT1), float(row.CDELT2), float(row.RSUN_REF)
            )
            x_mem[i - 1] = norm.numpy()
            raw_mem[i - 1] = raw.numpy()
            if i % 100 == 0 or i == len(records):
                print(f"prepared {i}/{len(records)}", flush=True)
        x_mem.flush()
        raw_mem.flush()
        del x_mem, raw_mem
        x_part.replace(x_path)
        raw_part.replace(raw_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return np.load(x_path, mmap_mode="r"), np.load(raw_path, mmap_mode="r")

    x_array, raw_array = prepare_cache(selected, Path(args.prepared_cache).resolve())
    positive_indices = selected.index[selected.label_m1plus_24h.eq(1)].astype(int).tolist()

    batch_size = int(checkpoint.get("batch_size", 16))
    physics_batch_size = int(checkpoint.get("physics_batch_size", 16))
    mixed_loader = DataLoader(
        PreparedDataset(selected, x_array, raw_array),
        batch_size=batch_size,
        sampler=mixed_sampler(selected, float(checkpoint.get("positive_fraction", 0.40)), seed),
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    physics_loader = DataLoader(
        PreparedDataset(selected, x_array, raw_array, positive_indices),
        batch_size=physics_batch_size,
        sampler=positive_group_sampler(positives, seed + 991),
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_channels = int(checkpoint.get("base_channels", 24))
    dropout = float(checkpoint.get("generator_dropout", 0.0))
    diffusion_steps = int(checkpoint.get("diffusion_steps", 100))
    model = ConditionalUNet(base=base_channels, dropout=dropout).to(device)
    ema = ConditionalUNet(base=base_channels, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    ema.eval()
    diffusion = Diffusion(diffusion_steps, device=device)

    lr = float(checkpoint.get("lr", 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_schedule = str(checkpoint.get("lr_schedule", "cosine"))
    eta_min = lr * 0.05

    def scheduled_lr(step: int) -> float:
        if lr_schedule != "cosine":
            return lr
        return eta_min + 0.5 * (lr - eta_min) * (
            1.0 + math.cos(math.pi * min(step, args.target_steps) / args.target_steps)
        )

    for group in opt.param_groups:
        group["lr"] = scheduled_lr(start_step)

    mix_rng = make_rng(device, seed + 100_003 + start_step)
    phys_rng = make_rng(device, seed + 200_003 + start_step)
    phys_cut = max(0, int(round((diffusion_steps - 1) * float(checkpoint.get("physics_max_t_frac", 0.25)))))
    physics_iter = iter(physics_loader)
    history: list[dict[str, float | int | bool]] = []
    step = start_step
    iterator = iter(mixed_loader)

    print(
        json.dumps(
            {
                "status": "RESUME_START",
                "start_step": start_step,
                "target_steps": args.target_steps,
                "remaining_steps": args.target_steps - start_step,
                "device": str(device),
                "torch_threads": torch.get_num_threads(),
                "checkpoint": str(checkpoint_path),
            }
        ),
        flush=True,
    )

    while step < args.target_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(mixed_loader)
            batch = next(iterator)

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lat = batch["lat"].to(device)
        raw = None
        t = torch.randint(0, diffusion_steps, (len(x),), device=device, generator=mix_rng)
        mixed_noise = randn_like_with(x, mix_rng)
        xt, noise = diffusion.q_sample(x, t, noise=mixed_noise)
        eps = model(xt, t, y, lat)
        denoise = torch.mean((eps - noise).square())

        lgeneric = denoise * 0.0
        lhj = denoise * 0.0
        lpil = denoise * 0.0
        if float(checkpoint.get("lambda_generic", 0.08)) > 0:
            pb, physics_iter = next_or_restart(physics_iter, physics_loader)
            px = pb["x"].to(device)
            raw = pb["raw"].to(device)
            plat = pb["lat"].to(device)
            pt = torch.randint(0, phys_cut + 1, (len(px),), device=device, generator=phys_rng)
            physics_noise = randn_like_with(px, phys_rng)
            pxt, _ = diffusion.q_sample(px, pt, noise=physics_noise)
            peps = model(pxt, pt, pb["y"].to(device), plat)
            px0hat = torch.clamp(diffusion.x0_from_eps(pxt, pt, peps), -1, 1)
            pfake = denormalize_gauss(px0hat)
            lgeneric = generic_distribution_loss(pfake, raw)

        step += 1
        ramp = min(1.0, step / float(checkpoint.get("physics_warmup_steps", 200)))
        total = denoise + ramp * float(checkpoint.get("lambda_generic", 0.08)) * lgeneric
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, float(checkpoint.get("ema_decay", 0.995)))
        new_lr = scheduled_lr(step)
        for group in opt.param_groups:
            group["lr"] = new_lr

        rec = {
            "step": step,
            "loss": float(total.item()),
            "denoise": float(denoise.item()),
            "generic": float(lgeneric.item()),
            "hj": float(lhj.item()),
            "pil": float(lpil.item()),
            "physics_ramp": float(ramp),
            "learning_rate": float(new_lr),
        }
        history.append(rec)
        if step % 20 == 0:
            print(json.dumps(rec), flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "condition": "base",
                    "seed": seed,
                    "steps": step,
                    "diffusion_steps": diffusion_steps,
                    "base_channels": base_channels,
                    "lambda_generic": float(checkpoint.get("lambda_generic", 0.08)),
                    "lambda_hj": float(checkpoint.get("lambda_hj", 0.05)),
                    "lambda_pil": float(checkpoint.get("lambda_pil", 0.05)),
                    "physics_max_t_frac": float(checkpoint.get("physics_max_t_frac", 0.25)),
                    "physics_batch_size": physics_batch_size,
                    "generator_dropout": dropout,
                    "ema_decay": float(checkpoint.get("ema_decay", 0.995)),
                    "lr_schedule": lr_schedule,
                    "device_at_save": str(device),
                    "resumed_from_step": start_step,
                    "optimizer_state_available": False,
                    "v2": True,
                },
                out / f"generator_step_{step}.pt",
            )

    final_state = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "condition": "base",
        "seed": seed,
        "steps": step,
        "diffusion_steps": diffusion_steps,
        "base_channels": base_channels,
        "lambda_generic": float(checkpoint.get("lambda_generic", 0.08)),
        "lambda_hj": float(checkpoint.get("lambda_hj", 0.05)),
        "lambda_pil": float(checkpoint.get("lambda_pil", 0.05)),
        "physics_max_t_frac": float(checkpoint.get("physics_max_t_frac", 0.25)),
        "physics_batch_size": physics_batch_size,
        "generator_dropout": dropout,
        "ema_decay": float(checkpoint.get("ema_decay", 0.995)),
        "lr_schedule": lr_schedule,
        "device_at_save": str(device),
        "resumed_from_step": start_step,
        "optimizer_state_available": False,
        "v2": True,
    }
    torch.save(final_state, out / "generator.pt")
    (out / "training_history_resume.json").write_text(json.dumps(history, indent=2) + "\n")
    (out / "run_config_resume.json").write_text(
        json.dumps(
            {
                "condition": "base",
                "seed": seed,
                "start_step": start_step,
                "target_steps": args.target_steps,
                "steps_completed": step,
                "remaining_steps": args.target_steps - start_step,
                "device": str(device),
                "torch_threads": torch.get_num_threads(),
                "source_dir": str(source_dir),
                "evidence_dir": str(evidence_dir),
                "fits_source": str(Path(args.fits_source).resolve()),
                "checkpoint": str(checkpoint_path),
                "optimizer_state_available": False,
                "note": "Resumed from a numbered model checkpoint; original optimizer/RNG state was not stored.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": "RESUME_COMPLETE",
                "condition": "base",
                "device": str(device),
                "steps": step,
                "checkpoint": str(out / "generator.pt"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
