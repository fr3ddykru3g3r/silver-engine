"""Aggregate five released-architecture comparator seeds and compare on identical rows."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs

SEEDS=(7,13,26,42,73)
CANDIDATES=("BASE_SOLAR","LATE_FUSION_SOLAR_XRS_PROTON")

def finite(v):
    if isinstance(v,dict): return {str(k):finite(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [finite(x) for x in v]
    if isinstance(v,np.ndarray): return [finite(x) for x in v.tolist()]
    if isinstance(v,np.generic): return finite(v.item())
    if isinstance(v,float) and not math.isfinite(v): return None
    return v

def save(path,obj): Path(path).write_text(json.dumps(finite(obj),indent=2,sort_keys=True,allow_nan=False)+"\n")

def pod80_threshold(y,p):
    r=v1.minimum_far_at_pod(y,p,.8)
    if r is None: raise ValueError("POD80 threshold unavailable")
    return float(r["threshold"])

def evaluate(y,p,t,roles,role,prevalence):
    m=roles==role
    return {
        **v1.threshold_metrics(y[m],p[m],t),
        **v1.probability_metrics(y[m],p[m],prevalence),
        "matched_detection":{str(x):v1.minimum_far_at_pod(y[m],p[m],x) for x in (.6,.7,.8,.9)},
        "rows":int(m.sum()),"positives":int(y[m].sum())
    }

def normalized_utc_times(values):
    """Normalize timestamps semantically, independent of pandas backing unit."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise")).as_unit("ns")

def run(seed_root:Path,late_csv:Path,output:Path):
    output=Path(output)
    if output.exists(): raise ValueError("output must be new")
    output.mkdir(parents=True)
    arrays=[]; receipts=[]
    ref=None
    for s in SEEDS:
        matches=list(Path(seed_root).rglob(f"seed_{s}.npz"))
        if len(matches)!=1: raise ValueError(f"expected one npz for seed {s}, got {matches}")
        # The object arrays are only string identifiers written by our own seed runner
        # (role/unit/time); model probabilities and labels remain numeric. The artifacts
        # are GitHub Actions outputs from the pinned seed jobs, not untrusted uploads.
        z=np.load(matches[0], allow_pickle=True)
        cur={k:z[k] for k in z.files}
        if ref is None: ref=cur
        else:
            for k in ("label","role","unit_id","issue_time"):
                if not np.array_equal(ref[k],cur[k]): raise ValueError(f"seed alignment mismatch {s} {k}")
        arrays.append(cur["probability"].astype(float))
        js=list(Path(seed_root).rglob(f"seed_{s}.json"))
        if len(js)==1: receipts.append(json.loads(js[0].read_text()))
    raw=np.median(np.stack(arrays),axis=0)
    y=ref["label"].astype(int); roles=ref["role"].astype(str); units=ref["unit_id"].astype(str)
    times=ref["issue_time"].astype(str)
    cal=roles=="calibration"; thr=roles=="threshold"; fit=roles=="fit"
    intercept=v1.fit_intercept(raw[cal],y[cal])
    prism=v1.sigmoid(v1.logit(raw)+intercept)
    prism_thresholds={
        "MAX_TSS":float(v1.select_threshold(y[thr],prism[thr])),
        "POD80_MIN_FAR":pod80_threshold(y[thr],prism[thr])
    }

    late=pd.read_csv(late_csv)
    late=late[late["scope"]=="ALL_HISTORY"].reset_index(drop=True)
    if len(late)!=len(y): raise ValueError("late-fusion row count mismatch")
    if not np.array_equal(late["label"].to_numpy(dtype=int),y): raise ValueError("late label mismatch")
    if not np.array_equal(late["role"].astype(str).to_numpy(),roles): raise ValueError("late role mismatch")
    lt=normalized_utc_times(late["issue_time"])
    rt=normalized_utc_times(times)
    if not lt.equals(rt): raise ValueError("late issue-time mismatch")
    if not np.array_equal(late["unit_id"].fillna("").astype(str).to_numpy(),units): raise ValueError("late unit mismatch")

    probs={"SEPNET_PRISM_RELEASED_ARCHITECTURE":prism}
    thresholds={"SEPNET_PRISM_RELEASED_ARCHITECTURE":prism_thresholds}
    for c in CANDIDATES:
        p=late[c].to_numpy(dtype=float)
        probs[c]=p
        thresholds[c]={
            "MAX_TSS":float(v1.select_threshold(y[thr],p[thr])),
            "POD80_MIN_FAR":pod80_threshold(y[thr],p[thr])
        }

    prevalence=float(np.mean(y[fit]))
    summary={
        "status":"COMPLETED_SAME_COHORT_RELEASED_ARCHITECTURE_COMPARATOR",
        "scope":"DEVELOPMENT_ONLY_ALREADY_INSPECTED_MONITOR",
        "locked_test_accessed":False,
        "target":v1.TARGET,
        "seed_aggregation":"median probability",
        "seeds":list(SEEDS),
        "prism_calibration_intercept":float(intercept),
        "models":{},"paired_monitor_comparisons":{},
        "seed_receipts":receipts,
    }
    for name,p in probs.items():
        summary["models"][name]={"thresholds":thresholds[name],"roles":{}}
        for policy,t in thresholds[name].items():
            summary["models"][name]["roles"][policy]={
                "score":evaluate(y,p,t,roles,"score",prevalence),
                "monitor":evaluate(y,p,t,roles,"monitor",prevalence),
            }

    for policy in ("MAX_TSS","POD80_MIN_FAR"):
        for other in CANDIDATES:
            key=f"SEPNET_PRISM_RELEASED_ARCHITECTURE_minus_{other}_{policy}"
            summary["paired_monitor_comparisons"][key]=cs.bootstrap_difference(
                y,prism,thresholds["SEPNET_PRISM_RELEASED_ARCHITECTURE"][policy],
                probs[other],thresholds[other][policy],units,roles,"monitor",
                seed=20260905,replicates=10000
            )

    out=pd.DataFrame({"issue_time":times,"role":roles,"unit_id":units,"label":y,
                      "SEPNET_PRISM_RELEASED_ARCHITECTURE":prism})
    for c in CANDIDATES: out[c]=probs[c]
    out.to_csv(output/"predictions.csv",index=False,float_format="%.17g")
    save(output/"summary.json",summary)
    save(output/"receipt.json",{
        "status":"DEVELOPMENT_ONLY_COMPARATOR_COMPLETE",
        "preregistration":"config/sepnet_prism_architecture_comparator_preregistration_2026-09-05.json",
        "locked_test_accessed":False,
        "monitor_prior_inspection_disclosed":True,
        "unfavorable_seeds_dropped":False,
        "timestamp_alignment_storage_unit_independent":True,
        "published_claim":"released architecture / fixed recipe under IRIS chronology; not exact paper random-IID reproduction"
    })

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed-root",type=Path,required=True)
    ap.add_argument("--late-fusion-predictions",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();run(a.seed_root,a.late_fusion_predictions,a.output)
if __name__=="__main__":main()
