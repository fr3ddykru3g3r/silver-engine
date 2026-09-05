from __future__ import annotations

"""Resumable positive-only v3 fidelity pilot.

The pilot is deliberately train-only. It writes an atomic ``latest.pt``
checkpoint and a JSON heartbeat before the hard workflow timeout so a stalled
or pre-empted runner leaves a usable audit trail. A checkpoint contains the
optimizer and RNG states as well as the model weights; the sampler order is
not promised to be bit-for-bit identical after resume.
"""

import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import MagnetogramDataset, build_records, cache_records
from generator import ConditionalUNet, Diffusion
from generic_fidelity import generic_distribution_loss
from physics_v2 import pil_distribution_loss_v2, population_distribution_loss_v2
from preprocess import denormalize_gauss


_STOP_REQUESTED = False


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "raw": torch.stack([b["raw_gauss"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "lat": torch.stack([b["latitude"] for b in batch]),
    }


def temporal_even(df: pd.DataFrame, n: int) -> pd.DataFrame:
    ordered = df.sort_values("t_rec").reset_index(drop=True)
    if len(ordered) <= n:
        return ordered.copy()
    indices = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(int))
    return ordered.iloc[indices[:n]].copy()


def positive_subset(full: pd.DataFrame, per_group: int) -> pd.DataFrame:
    positives = full[full.label_m1plus_24h.eq(1)].copy()
    parts = [temporal_even(group, per_group) for _, group in positives.groupby("region_group_id", sort=True)]
    if not parts:
        raise RuntimeError("No positive training records were found")
    return pd.concat(parts, ignore_index=True)


def region_sampler(df: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    sizes = df.groupby("region_group_id").size().to_dict()
    weights = np.asarray([1 / max(1, sizes[group]) for group in df.region_group_id], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(df), replacement=True, generator=generator)


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.data.mul_(decay).add_(parameter.data, alpha=1 - decay)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_write(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def checkpoint_payload(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float | int]],
    args: argparse.Namespace,
    step: int,
    device: torch.device,
    records: pd.DataFrame,
) -> dict[str, object]:
    return {
        "checkpoint_schema": 2,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "condition": args.condition,
        "seed": args.seed,
        "steps": step,
        "diffusion_steps": args.diffusion_steps,
        "base_channels": args.base_channels,
        "lambda_generic": args.lambda_generic,
        "lambda_hj": args.lambda_hj,
        "lambda_pil": args.lambda_pil,
        "physics_max_t_frac": args.physics_max_t_frac,
        "positive_rows": len(records),
        "positive_groups": int(records.region_group_id.nunique()),
        "device": str(device),
        "history": history,
    }


def write_heartbeat(out: Path, *, status: str, step: int, started: float, history: list[dict[str, float | int]], message: str = "") -> None:
    last = history[-1] if history else {}
    payload = {
        "status": status,
        "step": step,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "updated_unix": time.time(),
        "message": message,
        "last_metrics": last,
    }
    atomic_json_write(payload, out / "heartbeat.json")
    print(json.dumps({"heartbeat": payload}), flush=True)


def request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(json.dumps({"signal": "stop_requested", "message": "checkpointing after current batch"}), flush=True)


def load_resume(path: Path, model: torch.nn.Module, ema: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, list[dict[str, float | int]], dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    if checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    history = list(checkpoint.get("history", []))
    restore_rng_state(checkpoint.get("rng_state", {}))
    return int(checkpoint.get("steps", 0)), history, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", choices=["base", "hj", "pil", "hj_pil"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lambda-generic", type=float, default=0.08)
    parser.add_argument("--lambda-hj", type=float, default=0.10)
    parser.add_argument("--lambda-pil", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--physics-max-t-frac", type=float, default=0.20)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--heartbeat-every", type=int, default=10)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.checkpoint_every <= 0 or args.heartbeat_every <= 0:
        raise ValueError("checkpoint and heartbeat intervals must be positive")
    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)

    out = Path(args.out_dir) / args.condition
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    atomic_json_write({"status": "preparing", "started_unix": time.time(), "args": vars(args)}, out / "run_state.json")
    full = build_records(args.evidence_dir, "train")
    records = positive_subset(full, args.per_group)
    records = cache_records(records, Path(args.cache_dir), args.download_workers)
    dataset = MagnetogramDataset(records)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=region_sampler(records, args.seed), num_workers=0, collate_fn=collate, drop_last=True)

    model = ConditionalUNet(base=args.base_channels).to(device)
    ema = ConditionalUNet(base=args.base_channels).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    diffusion = Diffusion(args.diffusion_steps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cut = max(1, int((args.diffusion_steps - 1) * args.physics_max_t_frac))
    history: list[dict[str, float | int]] = []
    step = 0

    resume_path = args.resume
    if resume_path is not None:
        step, history, checkpoint = load_resume(resume_path, model, ema, optimizer)
        if checkpoint.get("condition") not in (None, args.condition):
            raise RuntimeError(f"resume condition mismatch: {checkpoint.get('condition')} != {args.condition}")
        if int(checkpoint.get("base_channels", args.base_channels)) != args.base_channels:
            raise RuntimeError("resume base_channels does not match the requested architecture")
        if int(checkpoint.get("diffusion_steps", args.diffusion_steps)) != args.diffusion_steps:
            raise RuntimeError("resume diffusion_steps does not match the requested schedule")
        model.to(device)
        ema.to(device)
        write_heartbeat(out, status="resumed", step=step, started=started, history=history, message=str(resume_path))
    else:
        write_heartbeat(out, status="training", step=step, started=started, history=history)

    atomic_json_write(
        vars(args) | {"device": str(device), "positive_rows": len(records), "positive_groups": int(records.region_group_id.nunique()), "resumed_from_step": step},
        out / "run_config.json",
    )

    while step < args.max_steps and not _STOP_REQUESTED:
        for batch in loader:
            if step >= args.max_steps or _STOP_REQUESTED:
                break
            x = batch["x"].to(device)
            raw = batch["raw"].to(device)
            labels = batch["y"].to(device)
            latitude = batch["lat"].to(device)
            t = torch.randint(0, args.diffusion_steps, (len(x),), device=device)
            noisy, noise = diffusion.q_sample(x, t)
            predicted = model(noisy, t, labels, latitude)
            denoise_loss = (predicted - noise).square().mean()

            physics_t = torch.randint(0, cut + 1, (len(x),), device=device)
            physics_noisy, physics_noise = diffusion.q_sample(x, physics_t)
            physics_predicted = model(physics_noisy, physics_t, labels, latitude)
            x0 = torch.clamp(diffusion.x0_from_eps(physics_noisy, physics_t, physics_predicted), -1, 1)
            fake = denormalize_gauss(x0)
            generic_loss = generic_distribution_loss(fake, raw)
            hj_loss = denoise_loss * 0
            pil_loss = denoise_loss * 0
            if args.condition in ("hj", "hj_pil"):
                hj_loss = population_distribution_loss_v2(fake, raw, latitude)
            if args.condition in ("pil", "hj_pil"):
                pil_loss = pil_distribution_loss_v2(fake, raw, 2.0)

            step += 1
            ramp = min(1.0, step / max(1, args.warmup_steps))
            loss = denoise_loss + ramp * (args.lambda_generic * generic_loss + args.lambda_hj * hj_loss + args.lambda_pil * pil_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_update(ema, model)
            metrics = {"step": step, "loss": float(loss.detach()), "denoise": float(denoise_loss.detach()), "generic": float(generic_loss.detach()), "hj": float(hj_loss.detach()), "pil": float(pil_loss.detach()), "ramp": ramp}
            history.append(metrics)
            print(json.dumps(metrics), flush=True)

            should_checkpoint = step % args.checkpoint_every == 0 or step >= args.max_steps or _STOP_REQUESTED
            should_heartbeat = step % args.heartbeat_every == 0 or should_checkpoint
            if should_checkpoint:
                payload = checkpoint_payload(model, ema, optimizer, history, args, step, device, records)
                atomic_torch_save(payload, out / "latest.pt")
                atomic_json_write({"status": "checkpointed", "step": step, "updated_unix": time.time(), "checkpoint": "latest.pt"}, out / "checkpoint_manifest.json")
            if should_heartbeat:
                write_heartbeat(out, status="checkpointed" if should_checkpoint else "training", step=step, started=started, history=history)
            if args.max_minutes > 0 and time.monotonic() - started >= args.max_minutes * 60:
                _STOP_REQUESTED = True
                print(json.dumps({"stop_reason": "max_minutes", "step": step}), flush=True)
        if args.max_minutes > 0 and time.monotonic() - started >= args.max_minutes * 60:
            _STOP_REQUESTED = True

    final_status = "completed" if step >= args.max_steps else "stopped_with_checkpoint"
    payload = checkpoint_payload(model, ema, optimizer, history, args, step, device, records)
    atomic_torch_save(payload, out / "generator.pt")
    atomic_torch_save(payload, out / "latest.pt")
    atomic_json_write({"status": final_status, "step": step, "updated_unix": time.time(), "checkpoint": "generator.pt", "history_rows": len(history)}, out / "checkpoint_manifest.json")
    atomic_json_write({"status": final_status, "finished_unix": time.time(), "steps": step, "resumed": resume_path is not None}, out / "run_state.json")
    atomic_json_write({"history": history}, out / "training_history.json")
    write_heartbeat(out, status=final_status, step=step, started=started, history=history)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"fatal_error": type(exc).__name__, "message": str(exc)}), flush=True)
        raise
