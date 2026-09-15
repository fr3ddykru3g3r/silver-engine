"""Independent, prediction-only audit; no training or network calls.

AI-assisted audit code, 2026-09-15. See audit_20260915/README.md for disclosure.
Historical files are accepted only through their immutable ZIP hash. Date
filtering precedes interpretation of labels, probabilities, and episode IDs.
The original artifact includes boundary-crossing targets, so full-cohort
verification is deliberately unavailable here. No frozen result is replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

BOUNDARY = datetime(2025, 9, 10, tzinfo=timezone.utc)
REPLAY_SHA = "81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0"
MODEL_FREE_SHA = "4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f"
MEMBER = "matched_episode_sensitivity_predictions.csv"


def safe_rows(stream, timestamp_column="issue_timestamp", *, reject_any=False):
    """Select solely on issue time; never inspect excluded-row outcomes.

    With inclusive horizon semantics, horizon_end == BOUNDARY is inadmissible.
    No excluded counts, labels, probabilities, IDs or timestamps are returned.
    This is an input-selection guard, not a claim about custody in earlier runs.
    """
    reader = csv.DictReader(stream)
    if not reader.fieldnames or timestamp_column not in reader.fieldnames:
        raise ValueError("missing issue timestamp")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("duplicate CSV columns")
    selected = []
    for row in reader:
        try:
            issue = datetime.fromisoformat(row[timestamp_column].replace("Z", "+00:00"))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid issue timestamp") from exc
        if issue.tzinfo is None:
            raise ValueError("timezone required")
        if issue + timedelta(hours=24) >= BOUNDARY:
            if reject_any:
                raise ValueError("protected horizon boundary: outcome inspection forbidden")
            continue
        selected.append(row)
    if not selected:
        raise ValueError("no admissible pre-boundary rows")
    return selected


def pinned_zip(path, expected):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("immutable archive hash mismatch")
    return zipfile.ZipFile(path)


def validate(frame):
    required = {"model", "issue_timestamp", "alert", "probability", "threshold",
                "standard_label", "episode_id", "eligibility", "quiet_block", "fold"}
    if not required.issubset(frame.columns):
        raise ValueError("incomplete prediction schema")
    f = frame.copy()
    for col in ("alert", "standard_label"):
        values = pd.to_numeric(f[col], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError("labels and alerts must be binary")
        f[col] = values.astype(int)
    for col in ("probability", "threshold"):
        f[col] = pd.to_numeric(f[col], errors="raise")
        if not np.isfinite(f[col]).all() or not f[col].between(0, 1).all():
            raise ValueError("probability/threshold must be finite and in [0,1]")
    if not np.array_equal(f.alert, (f.probability >= f.threshold).astype(int)):
        raise ValueError("stored alert disagrees with frozen probability/threshold")
    f["issue_timestamp"] = pd.to_datetime(f.issue_timestamp, utc=True)
    if (f.issue_timestamp + pd.Timedelta(hours=24) >= pd.Timestamp(BOUNDARY)).any():
        raise ValueError("protected horizon boundary")
    if f.duplicated(["model", "issue_timestamp"]).any():
        raise ValueError("duplicate model/issue")
    models = sorted(f.model.unique())
    if len(models) != 6:
        raise ValueError("six frozen comparators required")
    identity = ["issue_timestamp", "standard_label", "episode_id", "eligibility", "quiet_block", "fold"]
    base = f[f.model == models[0]].sort_values("issue_timestamp")[identity].reset_index(drop=True)
    for model in models:
        g = f[f.model == model].sort_values("issue_timestamp")[identity].reset_index(drop=True)
        if not g.equals(base):
            raise ValueError("models have different evaluation identities")
    for label, g in f.groupby("standard_label"):
        col = "episode_id" if label else "quiet_block"
        if g[col].isna().any() or g[col].astype(str).str.strip().eq("").any():
            raise ValueError("missing physical unit ID")
    if set(base.standard_label) != {0, 1}:
        raise ValueError("both classes required")
    return f


def covariance_gap(n, r, baseline_mass=None):
    """Finite-population covariance (ddof=0); arbitrary nonnegative base mass."""
    n, r = np.asarray(n, float), np.asarray(r, float)
    if n.ndim != 1 or n.size == 0 or n.shape != r.shape:
        raise ValueError("nonempty equal one-dimensional inputs required")
    if not np.isfinite(n).all() or not np.isfinite(r).all() or (n <= 0).any():
        raise ValueError("finite responses and positive multiplicities required")
    w = np.ones_like(n) if baseline_mass is None else np.asarray(baseline_mass, float)
    if w.shape != n.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("invalid baseline mass")
    w = w / w.sum()
    mn, mr = np.dot(w, n), np.dot(w, r)
    cov = np.dot(w, (n - mn) * (r - mr))
    return float(np.dot(w * n, r) / mn - mr), float(cov / mn)


def metrics(y, alert, p, w=None):
    y, a, p = np.asarray(y, int), np.asarray(alert, bool), np.asarray(p, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    tp, fn = w[(y == 1) & a].sum(), w[(y == 1) & ~a].sum()
    fp, tn = w[(y == 0) & a].sum(), w[(y == 0) & ~a].sum()
    if tp + fn == 0 or fp + tn == 0:
        raise ValueError("metric requires positive and negative mass")
    s, f = tp / (tp + fn), fp / (fp + tn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return dict(tp=float(tp), fn=float(fn), fp=float(fp), tn=float(tn), pod=float(s),
                fpr=float(f), far=float(fp / (tp + fp)) if tp + fp else None,
                tss=float(s-f), hss=float(2*(tp*tn-fn*fp)/den) if den else None,
                brier=float(np.average((p-y)**2, weights=w)), positive_mass=float(tp+fn),
                negative_mass=float(fp+tn), rows=len(y))


def view_metrics(g):
    pos = g.standard_label.eq(1)
    sizes = g[pos].groupby("episode_id").size()
    w = np.ones(len(g))
    w[pos] = 1 / g.loc[pos, "episode_id"].map(sizes).to_numpy(float)
    onset = g.eligibility.isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])
    return {
        "MAPPED_STANDARD": metrics(g.standard_label, g.alert, g.probability),
        "EPISODE_NORMALIZED_OCCURRENCE": metrics(g.standard_label, g.alert, g.probability, w),
        "NEW_ONSET": metrics(g.loc[onset, "eligibility"].eq("ELIGIBLE_ONSET_POSITIVE"),
                             g.loc[onset, "alert"], g.loc[onset, "probability"]),
    }


def mechanism(g):
    pos = g[g.standard_label == 1]
    grouped = pos.groupby("episode_id", sort=True)
    n = grouped.size().to_numpy(float)
    r = grouped.alert.mean().to_numpy(float)
    onset = pos[pos.eligibility == "ELIGIBLE_ONSET_POSITIVE"]
    k = onset.groupby("episode_id").size().reindex(grouped.size().index, fill_value=0).to_numpy(float)
    if not np.all(k == 1):
        raise ValueError("matched decomposition requires one onset issue per episode")
    u = onset.set_index("episode_id").alert.reindex(grouped.size().index).to_numpy(float)
    if not pos.eligibility.isin(["ELIGIBLE_ONSET_POSITIVE", "ALREADY_ACTIVE_PERSISTENCE"]).all():
        raise ValueError("unmodeled positive state")
    active_hits = grouped.alert.sum().to_numpy(float) - u
    v = np.divide(active_hits, n-1, out=np.zeros_like(n), where=n>1)
    gap, identity = covariance_gap(n, r)
    persistence = float(np.mean((1-1/n)*(v-u)))
    point = view_metrics(g)
    total_gap = point["MAPPED_STANDARD"]["tss"] - point["NEW_ONSET"]["tss"]
    negative_shift = point["NEW_ONSET"]["fpr"] - point["MAPPED_STANDARD"]["fpr"]
    tv = float(np.abs(n/n.sum()-1/len(n)).sum()/2)
    cs = float(np.std(n)*np.std(r)/np.mean(n))
    neg_scores = np.sort(g.loc[g.standard_label == 0, "probability"].to_numpy(float))
    probs = pos.probability.to_numpy(float)
    win = (np.searchsorted(neg_scores, probs, side="left") + np.searchsorted(neg_scores, probs, side="right"))/(2*len(neg_scores))
    auc_by_episode = pd.Series(win, index=pos.index).groupby(pos.episode_id).mean().reindex(grouped.size().index).to_numpy()
    auc_gap, auc_identity = covariance_gap(n, auc_by_episode)
    loss = (1-pos.probability)**2
    loss_ep = loss.groupby(pos.episode_id).mean().reindex(grouped.size().index).to_numpy()
    loss_gap, _ = covariance_gap(n, loss_ep)
    negative_loss = float(np.mean(g.loc[g.standard_label == 0, "probability"]**2))
    pi_row = len(pos)/len(g)
    pi_ep = len(n)/(len(n)+len(g)-len(pos))
    brier_reconstructed = pi_row*loss_gap + (pi_row-pi_ep)*(np.mean(loss_ep)-negative_loss)
    brier_gap = point["MAPPED_STANDARD"]["brier"]-point["EPISODE_NORMALIZED_OCCURRENCE"]["brier"]
    weights = []
    for alpha in [0, .25, .5, .75, 1]:
        mass = n**(1-alpha)
        weights.append(dict(alpha=alpha, pod=float(np.average(r, weights=mass)),
                            tss=float(np.average(r, weights=mass)-point["MAPPED_STANDARD"]["fpr"])))
    errors = [abs(gap-identity), abs(total_gap-gap-persistence-negative_shift), abs(auc_gap-auc_identity), abs(brier_gap-brier_reconstructed)]
    if max(errors)>1e-12 or abs(gap)>min(tv,cs)+1e-12:
        raise ValueError("exact decomposition/bound failed")
    return dict(episodes=len(n), positive_windows=len(pos), mean_multiplicity=float(n.mean()),
                multiplicity_gap=gap, covariance=float(identity*n.mean()), persistence_term=persistence,
                negative_cohort_term=negative_shift, mapped_minus_onset=total_gap,
                total_variation_bound=tv, cauchy_schwarz_bound=cs,
                multiplicity_cv=float(n.std()/n.mean()),
                kish_episode_weight_concentration=float(n.sum()**2/np.sum(n*n)),
                episode_any_alert_fraction=float(grouped.alert.max().mean()),
                onset_detection_fraction=float(u.mean()),
                mean_episode_window_detection=float(r.mean()),
                row_auc=float(np.average(auc_by_episode, weights=n)), episode_auc=float(auc_by_episode.mean()),
                auc_gap=auc_gap, brier_gap=brier_gap,
                brier_positive_reweighting_term=float(pi_row*loss_gap),
                brier_prior_shift_term=float((pi_row-pi_ep)*(np.mean(loss_ep)-negative_loss)),
                maximum_algebra_error=max(errors), weighting_path=weights)


def bootstrap(g, positive_indices, negative_indices, positive_ids, negative_ids):
    """Four-column physical-unit sufficient statistics, independent of runner."""
    out = {}
    pos = g.standard_label.eq(1)
    sizes = g[pos].groupby("episode_id").size()
    for name in ["MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"]:
        u = g.copy()
        u["weight"] = 1.0
        if name == "EPISODE_NORMALIZED_OCCURRENCE":
            u.loc[pos, "weight"] = 1/u.loc[pos,"episode_id"].map(sizes)
        if name == "NEW_ONSET":
            u = u[u.eligibility.isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
            u["standard_label"] = u.eligibility.eq("ELIGIBLE_ONSET_POSITIVE").astype(int)
        p = u[u.standard_label == 1].copy(); n = u[u.standard_label == 0].copy()
        p["hit"] = p.weight*p.alert; p["miss"] = p.weight*(1-p.alert)
        n["false"] = n.weight*n.alert; n["correct"] = n.weight*(1-n.alert)
        pc = p.groupby("episode_id")[["hit","miss"]].sum().reindex(positive_ids, fill_value=0).to_numpy()
        nc = n.groupby("quiet_block")[["false","correct"]].sum().reindex(negative_ids, fill_value=0).to_numpy()
        # Batch to bound memory, preserving each full shared resample.
        dist=[]
        for j in range(0,len(positive_indices),250):
            ps=pc[positive_indices[j:j+250]].sum(axis=1); ns=nc[negative_indices[j:j+250]].sum(axis=1)
            dist.extend(ps[:,0]/ps.sum(axis=1)-ns[:,0]/ns.sum(axis=1))
        out[name]=np.array(dist)
    return out


def interval(d):
    return dict(median=float(np.median(d)), interval95=np.quantile(d,[.025,.975]).tolist(),
                interval98333=np.quantile(d,[.05/6,1-.05/6]).tolist(),
                bootstrap_sign_fraction_negative=float(np.mean(d<0)),
                sign_fraction_is_not_p_value=True)


def paired(frame, draws=10000, seed=20260915):
    base=frame[frame.model == sorted(frame.model.unique())[0]]
    pi=sorted(base.loc[base.standard_label == 1,"episode_id"].unique())
    ni=sorted(base.loc[base.standard_label == 0,"quiet_block"].unique())
    rng=np.random.default_rng(seed)
    pdraw=rng.integers(0,len(pi),(draws,len(pi)))
    ndraw=rng.integers(0,len(ni),(draws,len(ni)))
    d={model:bootstrap(g,pdraw,ndraw,pi,ni) for model,g in frame.groupby("model")}
    contrasts={}
    for model in d:
        contrasts[model+"__normalized_minus_mapped"]=interval(d[model]["EPISODE_NORMALIZED_OCCURRENCE"]-d[model]["MAPPED_STANDARD"])
        contrasts[model+"__onset_minus_normalized"]=interval(d[model]["NEW_ONSET"]-d[model]["EPISODE_NORMALIZED_OCCURRENCE"])
    contrasts["proxy_onset_minus_mapped"]=interval(d["past_proton_ge10_proxy"]["NEW_ONSET"]-d["past_proton_ge10_proxy"]["MAPPED_STANDARD"])
    for family in ["xgb","elastic_net"]:
        contrasts[family+"__joint_minus_no_proton_onset"]=interval(d[family+"_joint"]["NEW_ONSET"]-d[family+"_no_proton"]["NEW_ONSET"])
    return dict(draws=draws,seed=seed,positive_units=len(pi),negative_units=len(ni),
                class_stratified=True,shared_across_models_and_views=True,
                newly_generated_audit_draws_not_original_frozen_bootstrap=True,contrasts=contrasts)


def run(archive, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with pinned_zip(archive,REPLAY_SHA) as z:
        # Verify contents cryptographically without interpreting any raw outcomes.
        manifest=json.loads(z.read("evidence_hashes.json"))
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:
                raise ValueError("artifact member hash mismatch")
        frame=validate(pd.DataFrame(safe_rows(io.TextIOWrapper(z.open(MEMBER)))))
    results=[];mechanisms={}
    for model,g in frame.groupby("model"):
        results.extend(dict(model=model,view=view,**m) for view,m in view_metrics(g).items())
        mechanisms[model]=mechanism(g)
    pd.DataFrame(results).to_csv(out/'strict_preboundary_metrics.csv',index=False)
    boot=paired(frame)
    robustness={}
    # Nonselective, descriptive checks; no chosen replacement threshold/model.
    for fold,g in frame.groupby("fold"):
        robustness['fold__'+fold]={m:mechanism(h) for m,h in g.groupby('model')}
    for days in [14,28]:
        f=frame.copy(); origin=pd.Timestamp('1970-01-05',tz='UTC')
        f['quiet_block']=((f.issue_timestamp-origin).dt.days//days).astype(str)
        robustness['quiet_block_days_'+str(days)]=paired(f)
    for scale in [.5,.75,1,1.25,1.5,2]:
        f=frame.copy(); f['threshold']=(f.threshold*scale).clip(0,1)
        f['alert']=(f.probability>=f.threshold).astype(int)
        # Only learned comparators; proxy/climatology are deterministic controls.
        robustness['threshold_scale_'+str(scale)]={m:mechanism(g) for m,g in f.groupby('model') if m.startswith(('xgb','elastic_net'))}
    # Whole event removal around temporal boundaries: selection by issue time only.
    f=frame.copy(); boundaries=pd.to_datetime(['2000-01-01','2005-01-01','2011-01-01','2018-01-01'],utc=True)
    near=np.zeros(len(f),bool)
    for b in boundaries:near|=(abs(f.issue_timestamp-b)<=pd.Timedelta(days=7)).to_numpy()
    affected=set(f.loc[near & f.standard_label.eq(1),'episode_id'])
    f=f[~near & ~f.episode_id.isin(affected)].copy()
    robustness['seven_day_role_edge_exclusion_no_refit']=paired(f)
    summary=dict(status='PASS_STRICT_PREBOUNDARY_DIAGNOSTIC',source_sha256=REPLAY_SHA,
                 audit_base_sha='13f692a90fc3006181d1c4488da78ff135f24f87',
                 cohort='fixed matched predictions with issue+24h strictly before 2025-09-10T00:00Z',
                 full_original_cohort_reverification='BLOCKED_BY_CONFLICTING_PROTECTED_BOUNDARY',
                 original_results_unchanged=True,models_refit=False,frozen_thresholds_changed=False,
                 robustness_thresholds_are_descriptive_only=True,admissible_issue_count=frame.issue_timestamp.nunique(),
                 mechanisms=mechanisms,bootstrap=boot,robustness=robustness,
                 protection='Excluded-row outcome fields are never interpreted; no protected cohort summary emitted.',
                 interpretation='Historical development-exposed diagnostic. Mathematical identities are not novelty claims.')
    (out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'output_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();r=run(a.archive,a.output)
    print(json.dumps({k:r[k] for k in ['status','admissible_issue_count','full_original_cohort_reverification']}))
