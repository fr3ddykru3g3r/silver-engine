from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import all_metrics, region_bootstrap
from .data import (
    FeatureScaler,
    SamplingConfig,
    TensorCacheDataset,
    attach_verified_fits,
    build_selected_records,
    collate,
    records_sha256,
    split_validation_roles,
    validate_evidence,
)
from .evaluation import (
    LogitCalibrator,
    atomic_json,
    choose_tss_threshold,
    ensemble_logits,
    paired_region_bootstrap_delta,
    physics_baseline,
    reliability_table,
    seed_all,
    sha256_file,
)
from .model import HybridFlareNet, parameter_count


@dataclass(frozen=True)
class TrainConfig:
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    width: int = 32
    dropout: float = 0.25
    epochs: int = 18
    patience: int = 5
    batch_size: int = 48
    workers: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 2e-4
    auxiliary_weight: float = 0.08
    gradient_clip: float = 5.0
    bootstrap_replicates: int = 5000


def _state(path: Path, phase: str, **extra) -> None:
    atomic_json(path, {"status": "RUNNING", "phase": phase, "updated_unix": time.time(), **extra})


def _loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate,
        drop_last=shuffle,
    )


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        output = model(batch["x"].to(device, non_blocking=True), batch["physics"].to(device, non_blocking=True))
        logits = output["logit"].detach().cpu().numpy()
        labels = batch["y"].numpy()
        rows.extend(
            {
                "sample_id": sample_id,
                "region_group_id": group,
                "y": int(label),
                "logit": float(logit),
            }
            for sample_id, group, label, logit in zip(batch["sample_id"], batch["group"], labels, logits)
        )
    return pd.DataFrame(rows)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    import random
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    import random
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_seed(
    seed: int,
    datasets: dict[str, TensorCacheDataset],
    config: TrainConfig,
    run_dir: Path,
    device: torch.device,
    state_path: Path,
) -> tuple[Path, list[dict]]:
    seed_all(seed)
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_path = seed_dir / "best.pt"
    latest_path = seed_dir / "latest.pt"
    train_loader = _loader(datasets["train"], config.batch_size, True, config.workers)
    validation_loader = _loader(datasets["validation_monitor"], config.batch_size, False, config.workers)
    model = HybridFlareNet(width=config.width, dropout=config.dropout).to(device)
    labels = datasets["train"].records.label_m1plus_24h.to_numpy(int)
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((1 - labels).sum()))
    positive_weight = math.sqrt(negatives / positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_score = -math.inf
    stale_epochs = 0
    history: list[dict] = []
    config_payload = asdict(config)
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != config_payload:
            raise RuntimeError(
                f"Checkpoint configuration mismatch for seed {seed}; use a new run directory"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
    if stale_epochs >= config.patience or start_epoch > config.epochs:
        if not best_path.is_file():
            raise RuntimeError(f"Seed {seed} resume has no best checkpoint")
        return best_path, history
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_classification = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device, non_blocking=True)
            physics = batch["physics"].to(device, non_blocking=True)
            target = batch["y"].to(device, non_blocking=True)
            auxiliary_target = batch["aux_target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with autocast:
                output = model(x, physics)
                classification_loss = criterion(output["logit"], target)
                auxiliary_loss = nn.functional.smooth_l1_loss(output["aux_physics"], auxiliary_target)
                loss = classification_loss + config.auxiliary_weight * auxiliary_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            total_classification += float(classification_loss.detach()) * len(target)
            seen += len(target)
            if batch_index % 25 == 0:
                _state(
                    state_path, "training", device=str(device), seed=seed,
                    epoch=epoch, batch=batch_index, samples_seen=seen,
                )
        scheduler.step()
        predictions = _predict(model, validation_loader, device)
        predictions["p"] = 1.0 / (1.0 + np.exp(-np.clip(predictions.logit, -50, 50)))
        validation_metrics = all_metrics(predictions.y, predictions.p, 0.5)
        score = float(validation_metrics["auprc"])
        record = {
            "epoch": epoch,
            "loss": total_loss / max(1, seen),
            "classification_loss": total_classification / max(1, seen),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_auprc": score,
            "validation_auroc": validation_metrics["auroc"],
            "validation_brier": validation_metrics["brier"],
        }
        history.append(record)
        print(json.dumps({"seed": seed, **record}), flush=True)
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint_payload = {
            "schema": 2,
            "seed": seed,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "stale_epochs": stale_epochs,
            "history": history,
            "rng_state": _rng_state(),
            "config": config_payload,
            "validation_auprc": score,
        }
        if improved:
            _save_checkpoint(best_path, checkpoint_payload)
        _save_checkpoint(latest_path, checkpoint_payload)
        atomic_json(seed_dir / "history.json", {"seed": seed, "history": history})
        _state(
            state_path, "training", device=str(device), seed=seed,
            epoch=epoch, best_validation_auprc=best_score,
            stale_epochs=stale_epochs,
        )
        if stale_epochs >= config.patience:
            break
    if not best_path.is_file():
        raise RuntimeError(f"Seed {seed} produced no best checkpoint")
    return best_path, history


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "pipeline_state.json"
    sampling = SamplingConfig(seed=args.seed)
    train_config = TrainConfig(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        workers=args.workers,
        bootstrap_replicates=args.bootstrap,
    )
    if args.smoke:
        train_config = TrainConfig(
            seeds=(args.seed,), width=16, epochs=1, patience=1, batch_size=16,
            workers=0, bootstrap_replicates=100,
        )
    _state(state_path, "evidence_contract")
    evidence_report = validate_evidence(args.evidence_dir, sampling.temporal_buffer_hours)
    frames = build_selected_records(args.evidence_dir, sampling)
    validation_roles = split_validation_roles(frames.pop("validation"), args.seed)
    frames.update(validation_roles)
    if args.smoke:
        for offset, name in enumerate(tuple(frames)):
            groups = sorted(frames[name].region_group_id.astype(str).unique())[: (48 if name == "train" else 24)]
            frames[name] = frames[name][frames[name].region_group_id.astype(str).isin(groups)].reset_index(drop=True)
    _state(state_path, "fits_contract")
    frames = attach_verified_fits(frames, args.fits_dir)
    scaler = FeatureScaler.fit(frames["train"])
    atomic_json(out / "feature_scaler.json", scaler.to_dict())
    data_contract = {
        "status": "PASS",
        "evidence": evidence_report,
        "sampling": asdict(sampling),
        "partitions": {
            name: {
                "rows": len(frame),
                "groups": int(frame.region_group_id.nunique()),
                "positive_rows": int(frame.label_m1plus_24h.sum()),
                "records_sha256": records_sha256(frame),
            }
            for name, frame in frames.items()
        },
    }
    atomic_json(out / "data_contract.json", data_contract)
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}_records.csv.gz", index=False, compression="gzip")
    _state(state_path, "tensor_cache")
    datasets = {
        name: TensorCacheDataset(frame, scaler, Path(args.tensor_cache) / name)
        for name, frame in frames.items()
    }
    if args.prewarm_cache:
        for name, dataset in datasets.items():
            for index in range(len(dataset)):
                dataset[index]
                if (index + 1) % 250 == 0 or index + 1 == len(dataset):
                    print(f"tensor cache {name}: {index + 1}/{len(dataset)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state(state_path, "training", device=str(device))
    checkpoints: list[Path] = []
    histories = {}
    for seed in train_config.seeds:
        checkpoint, history = _train_seed(seed, datasets, train_config, out, device, state_path)
        checkpoints.append(checkpoint)
        histories[str(seed)] = history
    _state(state_path, "validation_calibration")
    calibration_loader = _loader(datasets["validation_calibration"], train_config.batch_size, False, train_config.workers)
    threshold_loader = _loader(datasets["validation_threshold"], train_config.batch_size, False, train_config.workers)
    calibration_logits: list[pd.DataFrame] = []
    threshold_logits: list[pd.DataFrame] = []
    models: list[HybridFlareNet] = []
    for seed, checkpoint_path in zip(train_config.seeds, checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = HybridFlareNet(width=train_config.width, dropout=train_config.dropout).to(device)
        model.load_state_dict(checkpoint["model"])
        calibration_raw = _predict(model, calibration_loader, device)
        threshold_raw = _predict(model, threshold_loader, device)
        calibration_raw.to_csv(out / f"seed_{seed}" / "calibration_logits.csv", index=False)
        threshold_raw.to_csv(out / f"seed_{seed}" / "threshold_logits.csv", index=False)
        calibration_logits.append(calibration_raw)
        threshold_logits.append(threshold_raw)
        models.append(model)
    calibration_ensemble = ensemble_logits(calibration_logits)
    calibrator = LogitCalibrator()
    calibrator.fit(calibration_ensemble.logit.to_numpy(), calibration_ensemble.y.to_numpy())
    validation = ensemble_logits(threshold_logits)
    validation["p"] = calibrator.probabilities(validation.logit.to_numpy())
    threshold, validation_metrics = choose_tss_threshold(validation)
    validation.to_csv(out / "validation_ensemble.csv", index=False)
    atomic_json(
        out / "validation_selection.json",
        {
            "threshold": threshold,
            "metrics": validation_metrics,
            "calibrator": {"temperature": calibrator.temperature, "bias": calibrator.bias},
            "mean_seed_std": float(validation.p_std.mean()),
        },
    )
    _state(state_path, "locked_test_evaluation")
    test_loader = _loader(datasets["test"], train_config.batch_size, False, train_config.workers)
    test_logits: list[pd.DataFrame] = []
    for seed, model in zip(train_config.seeds, models):
        raw = _predict(model, test_loader, device)
        raw.to_csv(out / f"seed_{seed}" / "test_logits.csv", index=False)
        test_logits.append(raw)
    test = ensemble_logits(test_logits)
    test["p"] = calibrator.probabilities(test.logit.to_numpy())
    test.to_csv(out / "test_ensemble.csv", index=False)
    test_metrics = all_metrics(test.y, test.p, threshold)
    test_bootstrap = region_bootstrap(
        test,
        n_boot=train_config.bootstrap_replicates,
        seed=args.seed + 100,
        threshold=threshold,
    )
    baseline, baseline_test = physics_baseline(
        scaler.transform(frames["train"]),
        frames["train"].label_m1plus_24h.to_numpy(int),
        scaler.transform(frames["validation_calibration"]),
        frames["validation_calibration"].label_m1plus_24h.to_numpy(int),
        scaler.transform(frames["validation_threshold"]),
        frames["validation_threshold"].label_m1plus_24h.to_numpy(int),
        scaler.transform(frames["test"]),
        test,
        args.seed,
    )
    test["p_baseline"] = baseline_test.p.to_numpy()
    test.to_csv(out / "test_ensemble.csv", index=False)
    paired_delta = paired_region_bootstrap_delta(
        test, threshold, baseline["threshold"], train_config.bootstrap_replicates, args.seed + 121
    )
    model_example = models[0]
    receipt = {
        "status": "COMPLETE",
        "experiment": "independent_real_data_hybrid_flare_benchmark",
        "synthetic_gate_bypassed": False,
        "synthetic_gate_dependency": "none; this real-only benchmark is architecturally independent",
        "claim_boundary": "A breakthrough may be claimed only from this frozen held-out receipt and its group bootstrap.",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model": {
            "name": "HybridFlareNet",
            "parameters": parameter_count(model_example),
            "input_channels": ["signed_B", "unsigned_B", "PIL_proxy"],
            "physics_features": list(scaler.names),
            "auxiliary_task": "image_to_log_usflux_and_log_r_value",
            "seeds": list(train_config.seeds),
            "calibrator": {"temperature": calibrator.temperature, "bias": calibrator.bias},
        },
        "sampling": asdict(sampling),
        "train_config": asdict(train_config),
        "data_contract_sha256": sha256_file(out / "data_contract.json"),
        "validation_threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_region_bootstrap": test_bootstrap,
        "mean_test_seed_std": float(test.p_std.mean()),
        "reliability": reliability_table(test),
        "physics_only_baseline": baseline,
        "delta_vs_physics_baseline": {
            metric: float(test_metrics[metric] - baseline["test"][metric])
            for metric in ("tss", "hss", "auroc", "auprc", "bss")
        },
        "paired_region_bootstrap_delta_vs_physics": paired_delta,
        "histories": histories,
    }
    atomic_json(out / "final_receipt.json", receipt)
    atomic_json(state_path, {"status": "COMPLETE", "phase": "complete", "receipt": str(out / "final_receipt.json")})
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--fits-dir", required=True)
    parser.add_argument("--tensor-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--prewarm-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
