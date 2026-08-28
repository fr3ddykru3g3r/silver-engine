from __future__ import annotations

"""Independent train-only hard Hale/Joy evaluator with solar-cycle parity.

Hale polarity ordering reverses from one 11-year solar cycle to the next.  The
older v2 geometry diagnostic canonicalized only by hemisphere, which mixes Cycle
24 and Cycle 25 orientations and therefore is not a valid Hale-law population
metric across a multi-cycle HMI archive.  This evaluator canonicalizes the hard
positive-to-negative bipole vector by BOTH hemisphere and cycle parity, while the
selection generator uses a separate smooth descriptor.

No forecast labels beyond the already-fixed positive-training subset are used.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data import build_records, cache_records, MagnetogramDataset
from train_generator_v2 import group_temporal_subset
from preprocess import denormalize_gauss
from evaluate_generator_v2 import energy_distance, robust_standardize, split_half_baseline

CYCLE25_START = pd.Timestamp('2019-12-01T00:00:00Z')


def cycle_sign(times) -> np.ndarray:
    t=pd.to_datetime(pd.Series(times),utc=True,errors='coerce')
    if t.isna().any(): raise RuntimeError('Missing/invalid observation time for cycle-aware Hale metric')
    # Absolute sign convention is arbitrary; only the reversal between cycles matters.
    return np.where(t >= CYCLE25_START, 1.0, -1.0).astype(np.float32)


def hard_cycle_geometry(b: torch.Tensor, lat: np.ndarray, times) -> np.ndarray:
    _,_,h,w=b.shape
    yy=torch.linspace(-1,1,h).view(1,h,1); xx=torch.linspace(-1,1,w).view(1,1,w)
    cyc=cycle_sign(times); rows=[]
    for i in range(len(b)):
        z=b[i,0]; pw=torch.where(z>=150.0,z,torch.zeros_like(z)); nw=torch.where(z<=-150.0,-z,torch.zeros_like(z))
        ps=float(pw.sum()); ns=float(nw.sum())
        if ps<=0 or ns<=0:
            rows.append([0.,0.,0.,0.,0.]); continue
        px=float((pw*xx).sum()/pw.sum()); py=float((pw*yy).sum()/pw.sum())
        nx=float((nw*xx).sum()/nw.sum()); ny=float((nw*yy).sum()/nw.sum())
        dx=px-nx; dy=py-ny; sep=max((dx*dx+dy*dy)**0.5,1e-8)
        hemi=1.0 if float(lat[i])>=0 else -1.0
        canonical=hemi*float(cyc[i])
        # Canonical vector makes normal Hale ordering align across hemispheres/cycles;
        # the second component then retains Joy-law tilt information.
        rows.append([canonical*dx/sep,canonical*dy/sep,np.log1p(sep),np.log1p((ps+ns)/(h*w)),1.0])
    return np.asarray(rows,np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True)
    ap.add_argument('--synthetic-manifest',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--real-per-group',type=int,default=3); ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    rec=build_records(args.evidence_dir,'train'); rec=rec[rec.label_m1plus_24h.eq(1)].copy()
    rec=group_temporal_subset(rec,args.real_per_group,args.real_per_group,args.seed)
    rec=cache_records(rec,Path(args.cache_dir),workers=12).reset_index(drop=True)
    ds=MagnetogramDataset(rec); real=torch.stack([ds[i]['raw_gauss'] for i in range(len(ds))]).float()
    rlat=rec.latitude_deg.to_numpy(np.float32); rtime=rec.t_rec.astype(str).to_numpy()

    mp=Path(args.synthetic_manifest); man=pd.read_csv(mp); arr=[]
    for raw in man.array_path:
        q=Path(str(raw));
        if not q.exists(): q=mp.parent/'arrays'/q.name
        arr.append(np.load(q).astype(np.float32))
    syn=denormalize_gauss(torch.from_numpy(np.stack(arr))[:,None])
    if 'source_t_rec' not in man.columns:
        # Safe fallback for older manifests: recover source timestamp by immutable sample_id.
        full=build_records(args.evidence_dir,'train')[['sample_id','t_rec']].drop_duplicates('sample_id')
        man=man.merge(full.rename(columns={'sample_id':'source_sample_id','t_rec':'source_t_rec'}),on='source_sample_id',how='left',validate='many_to_one')
    slat=man.latitude_deg.to_numpy(np.float32); stime=man.source_t_rec.astype(str).to_numpy()

    rg=hard_cycle_geometry(real,rlat,rtime); sg=hard_cycle_geometry(syn,slat,stime)
    rgs,sgs,med,scale=robust_standardize(rg,sg)
    dist=energy_distance(torch.from_numpy(sgs),torch.from_numpy(rgs))
    rr=split_half_baseline(rgs,args.seed+113)
    ref=max(rr['p90'],rr['median'],1e-6)
    report={
        'evaluator':'independent hard Hale/Joy geometry, canonicalized by hemisphere and solar-cycle parity',
        'cycle25_start_utc':str(CYCLE25_START),
        'hard_cycle_geometry_standardized_energy_distance':dist,
        'real_split_half_distance':rr,
        'distance_to_real_split_p90_ratio':dist/ref,
        'real_count':len(real),'synthetic_count':len(syn),
        'robust_median':med.tolist(),'robust_scale':scale.tolist(),
        'forecast_outcomes_accessed':False,
    }
    pd.DataFrame(rg,columns=['canonical_ux','canonical_uy','log_sep','log_strong_flux_density','has_bipole']).assign(sample_id=rec.sample_id.astype(str).values,region_group_id=rec.region_group_id.astype(str).values,t_rec=rtime).to_csv(out/'real_hard_cycle_geometry.csv',index=False)
    pd.DataFrame(sg,columns=['canonical_ux','canonical_uy','log_sep','log_strong_flux_density','has_bipole']).assign(synthetic_id=man.synthetic_id.astype(str).values,source_t_rec=stime).to_csv(out/'synthetic_hard_cycle_geometry.csv',index=False)
    (out/'cycle_hj_metrics.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
