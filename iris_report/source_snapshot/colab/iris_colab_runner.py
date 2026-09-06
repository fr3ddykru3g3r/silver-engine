"""Colab runner for the locked IRIS/ISEF generator and forecasting protocol.

The runner is deliberately fail-closed. It never starts downstream forecasting
unless the independent train-only BASE/L0 fidelity gate has passed, and it
never treats a failed generator gate as a scientific result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback
import zipfile

import numpy as np
import pandas as pd
try:
    import requests
except ImportError:  # The preflight-only path can still explain missing FITS locally.
    requests = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "iris-model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from fit_cache import configure_local_source, verify_local_cache
WORK_ROOT = Path(os.environ.get("IRIS_WORK_ROOT", "/content/iris_silver_engine"))
RUN_BASE = os.environ.get("IRIS_RUN_BASE", "1") == "1"
RUN_PHYSICS = os.environ.get("IRIS_RUN_PHYSICS", "0") == "1"
RUN_DOWNSTREAM = os.environ.get("IRIS_RUN_DOWNSTREAM", "0") == "1"
SEED = int(os.environ.get("IRIS_SEED", "2026"))
EVIDENCE_RUN_ID = int(os.environ.get("IRIS_EVIDENCE_RUN_ID", "32937498427"))
EVIDENCE_ARTIFACT = os.environ.get(
    "IRIS_EVIDENCE_ARTIFACT", "iris-historical-evidence-integrity"
)


def run_cmd(args: list[object], label: str) -> None:
    cmd = [str(x) for x in args]
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_root(base: Path) -> Path:
    hits = list(base.rglob("training_manifest.csv.gz"))
    if not hits:
        raise RuntimeError("No training_manifest.csv.gz found in evidence archive")
    roots = [p.parent.parent.parent for p in hits if p.parent.name == "derived"]
    if not roots:
        raise RuntimeError("Evidence manifest has an unexpected path")
    return roots[0]


def unpack_archive(archive: Path, staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        root = staging / "zip_contents"
        root.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        nested = list(root.rglob("*.tar.gz")) + list(root.rglob("*.tgz"))
        if nested:
            tar_root = staging / "tar_contents"
            tar_root.mkdir(exist_ok=True)
            with tarfile.open(nested[0]) as t:
                t.extractall(tar_root)
            return evidence_root(tar_root)
        return evidence_root(root)
    if tarfile.is_tarfile(archive):
        root = staging / "tar_contents"
        root.mkdir(exist_ok=True)
        with tarfile.open(archive) as t:
            t.extractall(root)
        return evidence_root(root)
    raise RuntimeError("Evidence upload is not a zip or tar archive")


def download_public_artifact() -> Path | None:
    if requests is None:
        print("requests is not installed; skipping public evidence download", flush=True)
        return None
    api = (
        "https://api.github.com/repos/fr3ddykru3g3r/silver-engine/"
        f"actions/runs/{EVIDENCE_RUN_ID}/artifacts?per_page=100"
    )
    try:
        r = requests.get(api, timeout=60, headers={"Accept": "application/vnd.github+json"})
    except requests.RequestException as exc:
        print("Could not contact GitHub artifact API:", exc, flush=True)
        return None
    if r.status_code != 200:
        print("GitHub artifact listing returned", r.status_code, flush=True)
        return None
    matches = [
        x for x in r.json().get("artifacts", []) if x.get("name") == EVIDENCE_ARTIFACT
    ]
    if not matches:
        print("The requested public evidence artifact was not found", flush=True)
        return None
    try:
        q = requests.get(matches[0]["archive_download_url"], timeout=300)
    except requests.RequestException as exc:
        print("Could not download GitHub artifact:", exc, flush=True)
        return None
    if q.status_code != 200:
        print("GitHub artifact download returned", q.status_code, flush=True)
        return None
    out = WORK_ROOT / "iris-historical-evidence-integrity.zip"
    out.write_bytes(q.content)
    print("Downloaded artifact", out, sha256_file(out), flush=True)
    return out


def get_evidence() -> Path:
    configured = os.environ.get("IRIS_EVIDENCE_DIR")
    if configured:
        root = Path(configured)
        if (root / "data/derived/training_manifest.csv.gz").is_file():
            return root

    existing = WORK_ROOT / "evidence"
    if (existing / "data/derived/training_manifest.csv.gz").is_file():
        return existing

    archive = download_public_artifact()
    if archive is None:
        try:
            from google.colab import files
        except ImportError as exc:
            raise RuntimeError(
                "Evidence artifact could not be downloaded and this is not a Colab session. "
                "Set IRIS_EVIDENCE_DIR or provide an evidence archive."
            ) from exc
        print("Upload the evidence artifact (.zip/.tar.gz) now.", flush=True)
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No evidence archive was uploaded")
        archive = Path("/content") / next(iter(uploaded))

    root = unpack_archive(archive, WORK_ROOT / "evidence_unpack")
    repair = REPO_ROOT / "iris-gate0-data/repair_historical_tai_manifest.py"
    run_cmd(
        [sys.executable, repair, "--evidence-dir", root],
        "TAI-to-UTC manifest repair",
    )
    return root


def preflight(evidence: Path) -> None:
    derived = evidence / "data/derived"
    receipt = json.loads((derived / "tai_repair_audit.json").read_text())
    if receipt.get("status") != "PASS":
        raise RuntimeError("TAI repair receipt is not PASS")
    manifest = pd.read_csv(derived / "training_manifest.csv.gz", low_memory=False)
    parts = {p: set(manifest.loc[manifest.partition.eq(p), "region_group_id"].astype(str))
             for p in ["train", "validation", "test"]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = parts[a] & parts[b]
        if overlap:
            raise RuntimeError(f"Active-region overlap between {a} and {b}: {len(overlap)}")
    print("receipt_status =", receipt.get("status"), flush=True)
    print("rows_by_partition =", manifest.groupby("partition").size().to_dict(), flush=True)
    print("groups_by_partition =", manifest.groupby("partition").region_group_id.nunique().to_dict(), flush=True)
    print("positive_groups_by_partition =", manifest[manifest.label_m1plus_24h.eq(1)].groupby("partition").region_group_id.nunique().to_dict(), flush=True)
    print("ACTIVE_REGION_DISJOINTNESS_PASS", flush=True)
    if RUN_BASE or RUN_PHYSICS or RUN_DOWNSTREAM:
        fits_source = os.environ.get("IRIS_FITS_SOURCE", "").strip()
        if not fits_source:
            raise RuntimeError(
                "Real FITS images are not configured. Run colab/acquire_sharp_fits.py "
                "in Colab and set IRIS_FITS_SOURCE to its Drive directory before training."
            )
        configure_local_source(fits_source)
        cache_report = verify_local_cache(
            evidence,
            fits_source,
            run_base=RUN_BASE,
            run_physics=RUN_PHYSICS,
            run_downstream=RUN_DOWNSTREAM,
            seed=SEED,
            strict=True,
            write_report=WORK_ROOT / "fits_cache_preflight.json",
        )
        print("fits_cache_preflight =", json.dumps(cache_report, sort_keys=True), flush=True)
        print("REAL_FITS_CACHE_PREFLIGHT_PASS", flush=True)


def generator_run(evidence: Path, tag: str, condition: str) -> Path:
    root = WORK_ROOT / "runs" / tag
    out = root / "outputs"
    run_cmd(
        [
            sys.executable, REPO_ROOT / "iris-model/train_generator_v2.py",
            "--evidence-dir", evidence, "--cache-dir", WORK_ROOT / "cache/generator",
            "--out-dir", out, "--condition", condition, "--seed", SEED,
            "--per-group", 4, "--positive-slots", 4, "--batch-size", 16,
            "--physics-batch-size", 16, "--base-channels", 24,
            "--diffusion-steps", 100, "--max-steps", 1200, "--lr", 1e-4,
            "--lr-schedule", "cosine", "--lambda-generic", 0.08,
            "--lambda-hj", 0.05, "--lambda-pil", 0.05,
            "--physics-warmup-steps", 200, "--ema-decay", 0.995,
            "--ema-warmup-steps", 100, "--download-workers", 16,
            "--checkpoint-every", 200,
        ],
        f"{tag} generator training",
    )
    checkpoint = out / f"{condition}/generator.pt"
    samples = root / f"samples/{condition}"
    run_cmd(
        [
            sys.executable, REPO_ROOT / "iris-model/sample_generator.py",
            "--checkpoint", checkpoint, "--evidence-dir", evidence,
            "--out-dir", samples, "--per-group", 2, "--max-groups", 64,
            "--batch-size", 8, "--sampling-steps", 50, "--seed", SEED,
        ],
        f"{tag} matched train-positive sampling",
    )
    audit = root / "audit"
    run_cmd(
        [
            sys.executable, REPO_ROOT / "iris-model/evaluate_generator_v2_fixed.py",
            "--evidence-dir", evidence, "--cache-dir", WORK_ROOT / f"cache/eval-{tag}",
            "--synthetic-manifest", samples / "synthetic_manifest.csv",
            "--out-dir", audit, "--real-per-group", 2, "--seed", SEED,
        ],
        f"{tag} independent train-only fidelity evaluation",
    )
    report = json.loads((audit / "v2_manipulation_metrics.json").read_text())
    print(tag, json.dumps({k: report.get(k) for k in [
        "generic_fidelity_gate_pass", "generic_distance_to_real_baseline_ratio",
        "synthetic_to_real_flux_median_ratio", "synthetic_to_real_active_area_median_ratio",
        "synthetic_to_real_diversity_ratio", "synthetic_saturation_fraction_abs_gt_2900G",
    ]}, indent=2), flush=True)
    return audit / "v2_manipulation_metrics.json"


def gate_pass(path: Path) -> bool:
    return bool(json.loads(path.read_text()).get("generic_fidelity_gate_pass", False))


def paired_tss_bootstrap(a: pd.DataFrame, b: pd.DataFrame, ta: float, tb: float,
                         n_boot: int, seed: int) -> dict:
    from metrics import all_metrics

    keys = ["sample_id", "region_group_id", "y"]
    z = a[keys + ["p"]].merge(
        b[["sample_id", "p"]], on="sample_id", how="inner", suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(z) != len(a) or len(z) != len(b):
        raise RuntimeError("Paired arms do not have identical frozen test identities")
    point = all_metrics(z.y, z.p_a, ta)["tss"] - all_metrics(z.y, z.p_b, tb)["tss"]
    groups = np.asarray(sorted(z.region_group_id.astype(str).unique()))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        draw = rng.choice(groups, size=len(groups), replace=True)
        pieces = [z[z.region_group_id.astype(str).eq(g)] for g in draw]
        q = pd.concat(pieces, ignore_index=True)
        vals.append(all_metrics(q.y, q.p_a, ta)["tss"] - all_metrics(q.y, q.p_b, tb)["tss"])
    x = np.asarray(vals, dtype=float)
    return {
        "point_delta_tss": float(point),
        "median_delta_tss": float(np.nanmedian(x)),
        "lo95": float(np.nanpercentile(x, 2.5)),
        "hi95": float(np.nanpercentile(x, 97.5)),
        "bootstrap_replicates": n_boot,
        "bootstrap_unit": "region_group_id",
    }


def downstream(evidence: Path, manifests: dict[str, Path], added: int) -> None:
    root = WORK_ROOT / "runs/downstream"
    specs = [
        ("R", "real", "none", 0, None),
        ("Rw", "real_weighted", "balanced", 0, None),
        ("D", "duplicate", "none", added, None),
        ("L0", "synthetic", "none", added, manifests["L0"]),
        ("L2", "synthetic", "none", added, manifests["L2"]),
        ("L3", "synthetic", "none", added, manifests["L3"]),
    ]
    for label, arm, weighting, count, source in specs:
        cmd = [
            sys.executable, REPO_ROOT / "iris-model/train_matched_augmentation.py",
            "--evidence-dir", evidence, "--cache-dir", WORK_ROOT / "cache/downstream",
            "--out-dir", root / label, "--arm", arm, "--class-weighting", weighting,
            "--augmentation-count", count, "--seed", SEED,
            "--train-per-group", 4, "--val-per-group", 6, "--pos-cap", 2,
            "--test-per-group", 6, "--test-pos-cap", 2, "--width", 48,
            "--dropout", 0.2, "--gamma", 1.5, "--lr", 3e-4, "--batch-size", 32,
            "--steps", 1200, "--eval-every", 300, "--validation-bootstrap", 1000,
            "--test-bootstrap", 5000, "--download-workers", 16, "--evaluate-test",
        ]
        if source is not None:
            cmd += ["--synthetic-manifest", source]
        run_cmd(cmd, f"downstream arm {label}")

    from metrics import all_metrics
    summary = []
    frames = {}
    thresholds = {}
    for label in ["R", "Rw", "D", "L0", "L2", "L3"]:
        d = root / label
        report = json.loads((d / "metrics.json").read_text())
        pred = pd.read_csv(d / "test_predictions.csv")
        frames[label] = pred
        thresholds[label] = float(report["validation_threshold"])
        summary.append({"arm": label, "test_groups": report.get("test_groups"),
                        "test_rows": report.get("test_items"),
                        **{f"test_{k}": v for k, v in report["test"].items()}})
    pd.DataFrame(summary).to_csv(root / "primary_metrics.csv", index=False)
    comparisons = {}
    for label, ref in [("Rw", "R"), ("D", "R"), ("L0", "D"), ("L2", "D"), ("L3", "D")]:
        comparisons[f"{label}_minus_{ref}"] = paired_tss_bootstrap(
            frames[label], frames[ref], thresholds[label], thresholds[ref], 5000, SEED + len(comparisons)
        )
    (root / "primary_paired_tss_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)


def _main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    evidence = get_evidence()
    preflight(evidence)
    (WORK_ROOT / "run_config.json").write_text(json.dumps({
        "source_root": str(REPO_ROOT), "source_revision": "see notebook preflight",
        "work_root": str(WORK_ROOT), "seed": SEED, "run_base": RUN_BASE,
        "run_physics": RUN_PHYSICS, "run_downstream": RUN_DOWNSTREAM,
        "evidence_dir": str(evidence),
        "fits_source": os.environ.get("IRIS_FITS_SOURCE", ""),
    }, indent=2) + "\n")

    base_report = WORK_ROOT / "runs/base/audit/v2_manipulation_metrics.json"
    if RUN_BASE:
        base_report = generator_run(evidence, "base", "base")
    elif not base_report.is_file():
        raise RuntimeError("RUN_BASE=0 but no existing BASE gate report was found")
    if not gate_pass(base_report):
        raise RuntimeError("BASE fidelity gate failed; L2/L3 and forecasting are blocked")
    print("BASE_FIDELITY_GATE_PASS", flush=True)

    manifests = {
        "L0": WORK_ROOT / "runs/base/samples/base/synthetic_manifest.csv",
    }
    if RUN_PHYSICS:
        generator_run(evidence, "l2", "hj")
        generator_run(evidence, "l3", "hj_pil")
    manifests.update({
        "L2": WORK_ROOT / "runs/l2/samples/hj/synthetic_manifest.csv",
        "L3": WORK_ROOT / "runs/l3/samples/hj_pil/synthetic_manifest.csv",
    })
    if RUN_DOWNSTREAM:
        for p in manifests.values():
            if not p.is_file():
                raise RuntimeError(f"Missing synthetic manifest: {p}")
        counts = {k: len(pd.read_csv(v)) for k, v in manifests.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Synthetic exposure mismatch: {counts}")
        downstream(evidence, manifests, counts["L0"])
    else:
        print("RUN_DOWNSTREAM=0; forecasting matrix not started", flush=True)


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        failure = WORK_ROOT / "run_error.txt"
        failure.write_text(traceback.format_exc())
        print(f"RUN_FAILED; detailed traceback saved to {failure}", file=sys.stderr, flush=True)
        raise
