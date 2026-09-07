"""Independent audit of the availability-conditioned fallback artifact.

This script intentionally does not import the result runner or availability
fallback implementation.  It recomputes confusion metrics, probability metrics
and the predeclared DEGRADED gate directly from predictions.csv + summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


POLICIES = ("MAX_TSS", "POD80_MIN_FAR")
STATE_FOR_MODALITY = {
    "XRS": "NO_XRS",
    "PROTON": "NO_PROTON",
    "XRS_AND_PROTON": "NO_XRS_OR_PROTON",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite(v) for v in value]
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def threshold_metrics(y, p, t):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); pred = p >= float(t)
    tp = int(np.sum((y == 1) & pred)); fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred)); tn = int(np.sum((y == 0) & ~pred))
    pod = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    far = fp / (tp + fp) if tp + fp else None
    tss = None if pod is None or fpr is None else pod - fpr
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / denom if denom else None
    return {"TP":tp,"FN":fn,"FP":fp,"TN":tn,"POD":pod,"FPR":fpr,"FAR":far,"TSS":tss,"HSS":hss}


def probability_metrics(y, p, prevalence):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    brier=float(np.mean((p-y)**2)); ref=float(np.mean((float(prevalence)-y)**2))
    ece=0.0; edges=np.linspace(0,1,11)
    for i in range(10):
        m=(p>=edges[i]) & (p < edges[i+1] if i<9 else p<=edges[i+1])
        if np.any(m): ece += float(np.mean(m)*abs(np.mean(p[m])-np.mean(y[m])))
    return {
        "BRIER":brier,
        "BRIER_SKILL":None if ref<=0 else float(1-brier/ref),
        "ECE":ece,
        "AUPRC":float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
        "AUROC":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
    }


def metrics(y,p,mask,t,prevalence):
    m=np.asarray(mask,dtype=bool)
    if not np.any(m): raise ValueError("empty audit mask")
    return {**threshold_metrics(y[m],p[m],t),**probability_metrics(y[m],p[m],prevalence),"rows":int(m.sum()),"positives":int(np.sum(y[m]))}


def close(a,b,tol=2e-12):
    if a is None or b is None: return a is None and b is None
    return abs(float(a)-float(b)) <= tol


def compare_dict(observed, expected, prefix, mismatches, fields):
    for field in fields:
        if field not in observed or field not in expected or not close(observed.get(field),expected.get(field)):
            mismatches.append({"path":f"{prefix}.{field}","summary":observed.get(field),"audit":expected.get(field)})


def run(result_dir: Path, output: Path):
    result_dir=Path(result_dir); output=Path(output)
    summary=json.loads((result_dir/"summary.json").read_text())
    receipt=json.loads((result_dir/"receipt.json").read_text())
    pred_path=result_dir/"predictions.csv"; df=pd.read_csv(pred_path)
    if receipt.get("predictions_sha256") != digest(pred_path): raise ValueError("prediction hash mismatch")
    if summary.get("predictions_sha256") != digest(pred_path): raise ValueError("summary prediction hash mismatch")
    if summary.get("locked_test_accessed") or summary.get("monitor_used"): raise ValueError("forbidden evaluation role recorded")
    if summary.get("imputation_used") or summary.get("reconstruction_used"): raise ValueError("fallback experiment must not impute/reconstruct")

    y=df["label"].to_numpy(dtype=int); roles=df["role"].astype(str).to_numpy(); score=roles=="score"
    prevalence=float(summary["states"]["FULL"]["whole_score"]["MAX_TSS"]["positives"])
    # Fit prevalence is not needed to threshold metrics, but Brier skill uses it.
    # Recover it from the summary's clean full-score Brier-skill relation is
    # undesirable; instead read the prereg-compatible value stored in FULL
    # stack calibration context if available, falling back to score prevalence
    # only for Brier-skill audit. Absolute Brier/ECE are the admission fields.
    fit_prevalence = None
    if "fit_prevalence" in summary:
        fit_prevalence=float(summary["fit_prevalence"])
    if fit_prevalence is None:
        fit_prevalence=float(np.mean(y[roles=="fit"]))

    probs={state:df[f"p_{state}"].to_numpy(dtype=float) for state in ("FULL","NO_XRS","NO_PROTON","NO_XRS_OR_PROTON")}
    if any(not np.isfinite(p).all() for p in probs.values()): raise ValueError("nonfinite fallback probability")

    mismatches=[]; eval_count=0
    fields=("TP","FN","FP","TN","POD","FPR","FAR","TSS","HSS","BRIER","ECE","AUPRC","AUROC","rows","positives")
    for key,scenario in summary["scenarios"].items():
        modality=scenario["modality"]; state=STATE_FOR_MODALITY[modality]
        affected=(df[f"event_{key}"].to_numpy(dtype=int)==1)|(df[f"quiet_{key}"].to_numpy(dtype=int)==1)
        if int(np.sum(df[f"event_{key}"].to_numpy(dtype=int))) != scenario["event_rows"]: mismatches.append({"path":f"{key}.event_rows"})
        if int(np.sum(df[f"quiet_{key}"].to_numpy(dtype=int))) != scenario["quiet_control_rows"]: mismatches.append({"path":f"{key}.quiet_rows"})
        for policy in POLICIES:
            cand_t=float(summary["states"][state]["thresholds"][policy])
            ref_t=float(summary["states"]["FULL"]["thresholds"][policy])
            cand_a=metrics(y,probs[state],affected,cand_t,fit_prevalence)
            ref_a=metrics(y,probs["FULL"],affected,ref_t,fit_prevalence)
            cand_w=metrics(y,probs[state],score,cand_t,fit_prevalence)
            ref_w=metrics(y,probs["FULL"],score,ref_t,fit_prevalence)
            stored=scenario["policies"][policy]
            compare_dict(stored["candidate_affected"],cand_a,f"{key}.{policy}.candidate_affected",mismatches,fields)
            compare_dict(stored["reference_affected"],ref_a,f"{key}.{policy}.reference_affected",mismatches,fields)
            compare_dict(stored["candidate_whole_score"],cand_w,f"{key}.{policy}.candidate_whole",mismatches,fields)
            compare_dict(stored["reference_whole_score"],ref_w,f"{key}.{policy}.reference_whole",mismatches,fields)
            eval_count += 1

        primary=scenario["policies"]["MAX_TSS"]
        ca=primary["candidate_affected"]; cw=primary["candidate_whole_score"]; rw=primary["reference_whole_score"]
        checks={
            "affected_TSS_positive": ca["TSS"] is not None and ca["TSS"] > 0.0,
            "affected_POD_at_least_half": ca["POD"] is not None and ca["POD"] >= 0.50,
            "whole_score_Brier_within_margin": cw["BRIER"]-rw["BRIER"] <= 0.01,
            "whole_score_ECE_within_margin": cw["ECE"]-rw["ECE"] <= 0.02,
            "probabilities_finite": True,
        }
        expected_pass=all(checks.values())
        gate=scenario["operator_gate"]
        if bool(gate["passed"]) != expected_pass or gate["permission"] != ("DEGRADED" if expected_pass else "ABSTAIN") or gate["checks"] != checks:
            mismatches.append({"path":f"{key}.operator_gate","summary":gate,"audit_checks":checks})

    # State-level permission must require all three duration scenarios to pass.
    for state,decision in summary["operator_state_decision"].items():
        relevant=[v["operator_gate"]["passed"] for v in summary["scenarios"].values() if v["availability_state"]==state]
        expected=len(relevant)==3 and all(relevant)
        if decision["all_three_durations_pass_degraded_candidate_gate"] != expected or decision["permission"] != ("DEGRADED" if expected else "ABSTAIN") or decision.get("normal_allowed") is not False:
            mismatches.append({"path":f"operator_state_decision.{state}","summary":decision,"audit_expected":expected})

    audit={
        "status":"PASSED" if not mismatches else "FAILED",
        "independent_implementation":True,
        "imports_result_runner":False,
        "predictions_sha256":digest(pred_path),
        "summary_sha256":digest(result_dir/"summary.json"),
        "scenario_count":len(summary["scenarios"]),
        "policy_evaluations":eval_count,
        "mismatch_count":len(mismatches),
        "mismatches":mismatches,
    }
    output.write_text(json.dumps(finite(audit),indent=2,sort_keys=True,allow_nan=False)+"\n")
    if mismatches: raise SystemExit(f"availability fallback audit failed with {len(mismatches)} mismatches")
    return audit


def main():
    p=argparse.ArgumentParser(); p.add_argument("--result-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); print(json.dumps(run(a.result_dir,a.output),indent=2,sort_keys=True))

if __name__=="__main__": main()
