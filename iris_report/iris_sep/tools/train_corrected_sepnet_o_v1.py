"""Corrected, auditable SEPNET-O V1 development adapter.

The faithful comparator retains the archived dense multi-task architecture and
row-weighted objective.  ``episode_balanced`` is a separate predeclared IRIS
experiment.  Both accept only the approved v6 development cohort and never a
test-like role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (
    apply_calibration, fit_intercept_calibration, probability_metrics,
    select_tss_threshold, sigmoid, threshold_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_processed" / "sepnet_v1_development_v6_dual_target.csv"
SOURCE_MANIFEST = ROOT / "receipts" / "sepnet_v1_development_v6_dual_target_manifest.json"
SOURCE_SHA256 = "cc5ea62ff0a8423b8b9e3c028487dd61704fdeb950e84d4968b278179b841d04"
MANIFEST_SHA256 = "72f92f18936e237e9817fc4b425f2ebfec0096356b7a91efd65fbcf6852f4052"
FEATURE_SCHEMA_SHA256 = "7bca82f223f1be0adbd8afc6e30aed238ed52b3bb2339a98fa9c9cbd944436b5"
SEEDS = (7, 13, 26, 42, 73)
ROLES = ("train", "validation_monitor", "validation_calibration", "validation_threshold")
GENERAL = "future_SEP_label"
OPERATIONAL = "future_Operational_SEP_label"
FLUX = "future_SEP_MaxFlux"
META = {"issue_id", "role", "unit_id", "window_begin", "window_end", GENERAL, OPERATIONAL, FLUX}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SEPNetDense(nn.Module):
    def __init__(self, input_dim: int = 98, dropout: float = 0.3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 16), nn.LayerNorm(16), nn.ReLU(),
        )
        self.reg_head = nn.Linear(16, 1)
        self.cls_head = nn.Linear(16, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(inputs)  # exactly one shared forward pass
        return self.reg_head(shared).squeeze(-1), self.cls_head(shared).squeeze(-1)


class Rows(Dataset):
    def __init__(self, indices: np.ndarray, x: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, weights: np.ndarray):
        self.indices, self.x, self.y_cls, self.y_reg, self.weights = indices, x, y_cls, y_reg, weights
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, position: int):
        index = int(self.indices[position])
        return (torch.from_numpy(self.x[index]), torch.tensor(self.y_cls[index], dtype=torch.float32),
                torch.tensor(self.y_reg[index], dtype=torch.float32), torch.tensor(self.weights[index], dtype=torch.float32),
                torch.tensor(index, dtype=torch.int64))


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _rng() -> dict[str, Any]:
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state: torch.cuda.set_rng_state_all(state["cuda"])


def _focal(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return 0.25 * (1.0 - torch.exp(-bce)).pow(2.0) * bce


def _loss(regression: torch.Tensor, logits: torch.Tensor, yreg: torch.Tensor, ycls: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    regression_loss = (regression - yreg).pow(2)
    classification_loss = nn.functional.binary_cross_entropy_with_logits(logits, ycls, reduction="none") + 10.0 * _focal(logits, ycls)
    return ((regression_loss + classification_loss) * weights).sum() / weights.sum()


def fit_preprocessing(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, Any], bytes]:
    train = frame["role"].eq("train").to_numpy()
    raw = frame[features].to_numpy(dtype=np.float64)
    observed = np.isfinite(raw)
    safe = raw.copy(); safe[~observed] = np.nan
    imputer = KNNImputer(n_neighbors=10, weights="uniform", keep_empty_features=True)
    train_imputed = imputer.fit_transform(safe[train])
    all_imputed = imputer.transform(safe)
    scaler = MinMaxScaler().fit(train_imputed)
    train_scaled = scaler.transform(train_imputed)
    all_scaled = scaler.transform(all_imputed)
    selector = SelectKBest(score_func=f_classif, k=98).fit(train_scaled, frame.loc[train, GENERAL].to_numpy(dtype=int))
    transformed = selector.transform(all_scaled).astype(np.float32)
    if transformed.shape[1] != 98 or not np.isfinite(transformed).all():
        raise ValueError("preprocessing must produce exactly 98 finite selected predictors")
    payload = pickle.dumps({"imputer": imputer, "scaler": scaler, "selector": selector}, protocol=5)
    receipt = {
        "fit_role": "train", "knn_imputer_neighbors": 10, "scaler": "MinMaxScaler",
        "selector": "SelectKBest(f_classif,k=98)", "ordered_input_features": features,
        "selected_feature_indices": selector.get_support(indices=True).tolist(),
        "missingness_preserved_for_audit": True,
        "missing_counts_by_feature": {name: int((~observed[:, i]).sum()) for i, name in enumerate(features)},
        "note": "Missingness is retained in the receipt; KNN values are fitted on train only and no outcome fills a feature.",
    }
    return transformed, receipt, payload


@torch.no_grad()
def _predict(model: SEPNetDense, loader: DataLoader, rows: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); logits = np.full(rows, np.nan); regression = np.full(rows, np.nan)
    for x, _, _, _, index in loader:
        reg, logit = model(x.to(device)); logits[index.numpy()] = logit.cpu().numpy(); regression[index.numpy()] = reg.cpu().numpy()
    return logits, regression


def run(mode: str, output_dir: Path, *, max_epochs: int = 150, patience: int = 20, resume: bool = False) -> dict[str, Any]:
    if mode not in {"faithful_row_weighted", "episode_balanced"}: raise ValueError("invalid predeclared mode")
    if output_dir.exists() and not resume: raise ValueError("output exists; use --resume only for the same run")
    if sha256_file(SOURCE) != SOURCE_SHA256 or sha256_file(SOURCE_MANIFEST) != MANIFEST_SHA256:
        raise ValueError("only the approved v6 dual-target cohort is accepted")
    source_receipt = json.loads(SOURCE_MANIFEST.read_text())
    if source_receipt["output_sha256"] != SOURCE_SHA256 or source_receipt["testing_or_sepval_artifact_accessed"] is not False:
        raise ValueError("v6 provenance contract failed")
    frame = pd.read_csv(SOURCE, float_precision="round_trip")
    if set(frame["role"]) != set(ROLES) or frame["role"].str.lower().str.contains("test|sepval|locked").any():
        raise ValueError("test-like or unexpected roles are forbidden")
    features = [column for column in frame.columns if column not in META]
    feature_hash = hashlib.sha256(json.dumps(features, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(features) != 98 or feature_hash != FEATURE_SCHEMA_SHA256 or any(name.lower().startswith("future_") for name in features):
        raise ValueError("feature schema mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"mode": mode, "source_sha256": SOURCE_SHA256, "seeds": list(SEEDS), "max_epochs": max_epochs, "patience": patience}
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists() and json.loads(run_config_path.read_text()) != run_config:
        raise ValueError("resume configuration differs from the interrupted run")
    if not run_config_path.exists():
        run_config_path.write_text(json.dumps(run_config, sort_keys=True, indent=2) + "\n")
    x, preprocessing, preprocessing_bytes = fit_preprocessing(frame, features)
    preprocessing_path = output_dir / "preprocessing.pkl"
    preprocessing_receipt_path = output_dir / "preprocessing_receipt.json"
    if not preprocessing_path.exists(): preprocessing_path.write_bytes(preprocessing_bytes)
    elif preprocessing_path.read_bytes() != preprocessing_bytes: raise ValueError("resume preprocessing mismatch")
    preprocessing_receipt_path.write_text(json.dumps(preprocessing, sort_keys=True, indent=2) + "\n")
    missingness_path = output_dir / "observed_feature_mask.npz"
    if not missingness_path.exists():
        np.savez_compressed(missingness_path, observed=np.isfinite(frame[features].to_numpy(dtype=np.float64)), features=np.asarray(features))
    general = frame[GENERAL].to_numpy(dtype=np.int64)
    operational = frame[OPERATIONAL].to_numpy(dtype=np.int64)
    yreg = np.log1p(frame[FLUX].to_numpy(dtype=np.float64)).astype(np.float32)
    indices = {role: np.flatnonzero(frame["role"].eq(role)) for role in ROLES}
    weights = np.ones(len(frame), dtype=np.float32)
    if mode == "episode_balanced":
        counts = frame.loc[indices["train"], "unit_id"].value_counts()
        weights[indices["train"]] = frame.loc[indices["train"], "unit_id"].map(lambda unit: 1.0 / counts[unit]).to_numpy()
        weights[indices["train"]] /= weights[indices["train"]].mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {role: DataLoader(Rows(rows, x, general, yreg, weights), batch_size=128, shuffle=(role == "train"), num_workers=0) for role, rows in indices.items()}
    logits_seeds, regression_seeds, seed_receipts = [], [], []
    started = time.time()
    for seed in SEEDS:
        _seed(seed); seed_dir = output_dir / f"seed_{seed}"; seed_dir.mkdir(exist_ok=True)
        model = SEPNetDense().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        last, best = seed_dir / "last.pt", seed_dir / "best.pt"
        start_epoch=0; best_loss=float("inf"); best_epoch=-1; stale=0
        if resume and last.exists():
            state=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
            start_epoch=state["epoch"]+1; best_loss=state["best_loss"]; best_epoch=state["best_epoch"]; stale=state["stale"]; _restore_rng(state["rng"])
        for epoch in range(start_epoch, max_epochs):
            model.train()
            for xb,yc,yr,w,_ in loaders["train"]:
                xb,yc,yr,w=xb.to(device),yc.to(device),yr.to(device),w.to(device); optimizer.zero_grad(set_to_none=True)
                reg,logit=model(xb); loss=_loss(reg,logit,yr,yc,w); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0); optimizer.step()
            model.eval(); monitor_total=0.0; monitor_weight=0
            with torch.no_grad():
                for xb,yc,yr,w,_ in loaders["validation_monitor"]:
                    xb,yc,yr=xb.to(device),yc.to(device),yr.to(device); unit=torch.ones_like(yc); reg,logit=model(xb)
                    batch_loss=_loss(reg,logit,yr,yc,unit); monitor_total += float(batch_loss)*len(yc); monitor_weight += len(yc)
            monitor_loss=monitor_total/monitor_weight; scheduler.step(monitor_loss)
            if monitor_loss < best_loss-1e-7:
                best_loss=monitor_loss; best_epoch=epoch; stale=0
                torch.save({"model":model.state_dict(),"epoch":epoch,"monitor_loss":monitor_loss},best.with_suffix(".tmp")); os.replace(best.with_suffix(".tmp"),best)
            else: stale += 1
            state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best_loss":best_loss,"best_epoch":best_epoch,"stale":stale,"rng":_rng()}
            torch.save(state,last.with_suffix(".tmp")); os.replace(last.with_suffix(".tmp"),last)
            if stale >= patience: break
        best_state=torch.load(best,map_location=device,weights_only=False); model.load_state_dict(best_state["model"])
        logits,regression=_predict(model,DataLoader(Rows(np.arange(len(frame)),x,general,yreg,weights),batch_size=512),len(frame),device)
        logits_seeds.append(logits); regression_seeds.append(regression)
        seed_receipts.append({"seed":seed,"best_epoch":best_epoch,"best_monitor_combined_loss":best_loss,"best_checkpoint":str(best.relative_to(output_dir)),"best_checkpoint_sha256":sha256_file(best),"last_checkpoint":str(last.relative_to(output_dir)),"last_checkpoint_sha256":sha256_file(last)})
    calibrated=[]; calibration_receipts=[]
    for seed,logits in zip(SEEDS,logits_seeds):
        calibration=fit_intercept_calibration(logits[indices["validation_calibration"]],operational[indices["validation_calibration"]],role="validation_calibration")
        calibrated.append(apply_calibration(logits,calibration)); calibration_receipts.append({"seed":seed,"intercept":calibration.intercept,"calibration_id":calibration.calibration_id,"fit_role":calibration.fit_role})
    probability=np.median(np.stack(calibrated),axis=0); regression_log=np.median(np.stack(regression_seeds),axis=0)
    threshold=select_tss_threshold(operational[indices["validation_threshold"]],probability[indices["validation_threshold"]],role="validation_threshold")
    prediction=pd.DataFrame({"issue_id":frame["issue_id"],"role":frame["role"],"unit_id":frame["unit_id"],"general_label":general,"operational_label":operational,"max_flux":frame[FLUX],"ensemble_probability":probability,"ensemble_log1p_flux_prediction":regression_log})
    prediction_path=output_dir/"development_predictions.csv"; prediction.to_csv(prediction_path,index=False,float_format="%.17g")
    reference=float(operational[indices["train"]].mean()); metrics={}
    for role,rows in indices.items(): metrics[role]={**probability_metrics(operational[rows],probability[rows],reference_probability=reference),**threshold_metrics(operational[rows],probability[rows],threshold.threshold)}
    archived=ROOT/"workstreams"/"luna_a"/"sources"/"sepnets_v1"/"multitask_model.py"
    receipt={"status":"PASS_DEVELOPMENT_ONLY","experiment_mode":mode,"comparator_role":"FAITHFUL_CORRECTED_SEPNET_O" if mode=="faithful_row_weighted" else "PREDECLARED_IRIS_EPISODE_BALANCED_EXPERIMENT","architecture":[98,256,128,64,16],"heads":["classification_general_SEP","regression_log1p_max_flux"],"one_shared_forward_pass":True,"training_labels":{"classification":GENERAL,"regression":f"log1p({FLUX})","operational_label_used_for_training":False},"role_contract":{"early_stopping":"validation_monitor_general_plus_regression_loss","calibration":"validation_calibration_operational_label","threshold":"validation_threshold_operational_label"},"weighting":"ROW_WEIGHTED" if mode=="faithful_row_weighted" else "INVERSE_UNIT_SIZE_TRAIN_ONLY","seeds":seed_receipts,"calibrations":calibration_receipts,"threshold":{"value":threshold.threshold,"fit_role":threshold.fit_role,"threshold_id":threshold.threshold_id},"metrics_operational_label":metrics,"runtime_seconds":time.time()-started,"device":str(device),"source_sha256":SOURCE_SHA256,"source_manifest_sha256":MANIFEST_SHA256,"feature_schema_sha256":feature_hash,"preprocessing_sha256":sha256_file(preprocessing_path),"preprocessing_receipt_sha256":sha256_file(preprocessing_receipt_path),"predictions_sha256":sha256_file(prediction_path),"source_code_sha256":{"adapter":sha256_file(Path(__file__)),"archived_model":sha256_file(archived)},"source_deviations":["fixed undefined val_loss/train_loss defect","one shared forward pass instead of two stochastic forwards","gradient clipping after backward instead of before backward","chronological four-role development protocol","intercept calibration and TSS threshold have dedicated roles","restart-safe checkpoints and five fixed seeds"],"locked_test_accessed":False,"headline_eligible_roles":[],"claims_forbidden":["SEPVAL_SCORE","FINAL_NEW_CROSSING_SCORE","SUPERIORITY","BREAKTHROUGH","OPERATIONAL_CERTIFICATION","PRODUCTION_READINESS"]}
    receipt_path=output_dir/"receipt.json"; receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=["faithful_row_weighted","episode_balanced"],required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--max-epochs",type=int,default=150); parser.add_argument("--patience",type=int,default=20); parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(); print(json.dumps(run(args.mode,args.output_dir,max_epochs=args.max_epochs,patience=args.patience,resume=args.resume),sort_keys=True))


if __name__=="__main__": main()
