from __future__ import annotations

"""Cycle-aware train-only Hale/Joy morphology selector.

Creates a fixed candidate pool per positive training magnetogram using the already
passed morphology-bootstrap BASE family. Candidate transforms alter only global
polarity/orientation degrees of freedom relevant to Hale/Joy population structure:
small rigid rotations and optional whole-map polarity reversal. Selection matches
the real smooth bipole-geometry distribution after canonicalizing by hemisphere
AND solar-cycle parity, while retaining a source-preservation penalty.

The final scientific manipulation decision is made by evaluate_cycle_aware_hj.py
(hard 150 G thresholds), not by this smooth selector. No forecast outcomes are used.
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import build_records, cache_records, MagnetogramDataset
from preprocess import normalize_gauss, denormalize_gauss
from train_generator_v2 import group_temporal_subset
from generate_residual_bootstrap import blur
from generate_morph_bootstrap import elastic_warp
from physics_v2 import polarity_geometry_descriptor_v2
from generate_morph_physics_selected import robust_center_scale, energy_distance_multivariate

CYCLE25_START=pd.Timestamp('2019-12-01T00:00:00Z')


def seed_all(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def cycle_sign(times):
    t=pd.to_datetime(pd.Series(times),utc=True,errors='coerce')
    if t.isna().any(): raise RuntimeError('invalid t_rec')
    return np.where(t>=CYCLE25_START,1.0,-1.0).astype(np.float32)

def rotate(x:torch.Tensor,deg:float)->torch.Tensor:
    if abs(deg)<1e-9: return x.clone()
    th=np.deg2rad(deg); c=float(np.cos(th)); s=float(np.sin(th))
    theta=torch.tensor([[[c,-s,0.0],[s,c,0.0]]],dtype=x.dtype,device=x.device)
    grid=F.affine_grid(theta,x.size(),align_corners=False)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=False)

def smooth_cycle_descriptor(raw:torch.Tensor,lat:np.ndarray,times)->np.ndarray:
    lt=torch.from_numpy(lat.astype(np.float32)); cyc=torch.from_numpy(cycle_sign(times)).to(raw.dtype)
    with torch.no_grad(): z=polarity_geometry_descriptor_v2(raw,lt).cpu().numpy()
    # v2 descriptor has hemisphere*ux, hemisphere*uy. Multiply both by cycle parity
    # so the polarity-defined vector is canonical across the 22-year Hale cycle.
    z=z.copy(); z[:,0]*=cyc.numpy(); z[:,1]*=cyc.numpy(); return z.astype(np.float32)

def objective(sel,cdesc,generic,target,lam):
    g=float(np.mean(generic[np.arange(len(sel)),sel])); z=cdesc[np.arange(len(sel)),sel]
    p=energy_distance_multivariate(z,target); return g+lam*p,g,p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--lambda-physics',type=float,required=True); ap.add_argument('--warp-px',type=float,default=2.0); ap.add_argument('--residual-alpha',type=float,default=0.15); ap.add_argument('--residual-sigma',type=float,default=4.0)
    ap.add_argument('--per-group',type=int,default=2); ap.add_argument('--max-groups',type=int,default=64); ap.add_argument('--passes',type=int,default=4); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--download-workers',type=int,default=16)
    args=ap.parse_args(); seed_all(args.seed); out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    rec=build_records(args.evidence_dir,'train'); rec=rec[rec.label_m1plus_24h.eq(1)].copy(); rec=group_temporal_subset(rec,args.per_group,args.per_group,args.seed)
    groups=sorted(rec.region_group_id.astype(str).unique())
    if args.max_groups and len(groups)>args.max_groups:
        rr=np.random.default_rng(args.seed); groups=sorted(rr.choice(groups,args.max_groups,replace=False).tolist()); rec=rec[rec.region_group_id.astype(str).isin(groups)].copy()
    rec=cache_records(rec,Path(args.cache_dir),args.download_workers).reset_index(drop=True); ds=MagnetogramDataset(rec)
    raw=torch.stack([ds[i]['raw_gauss'] for i in range(len(ds))]).float(); norm=torch.from_numpy(np.stack([normalize_gauss(z[0].numpy()) for z in raw]))[:,None].float()
    lat=rec.latitude_deg.to_numpy(float); times=rec.t_rec.astype(str).to_numpy(); gids=rec.region_group_id.astype(str).to_numpy()
    real_desc=smooth_cycle_descriptor(raw,lat,times); med,scale=robust_center_scale(real_desc); target=((real_desc-med)/scale).astype(np.float32)

    # Fixed intervention set. Polarity reversal is physically interpretable here as
    # the degree of freedom needed to reproduce normal vs anti-Hale population rates;
    # rotations adjust Joy-law tilt without changing field-strength marginals.
    transforms=[(d,f) for f in (1.0,-1.0) for d in (-12.,-8.,-4.,0.,4.,8.,12.)]
    n=len(rec); kmax=len(transforms); pool=[[None]*kmax for _ in range(n)]; donor=np.zeros((n,kmax),int); generic=np.zeros((n,kmax)); desc=np.zeros((n,kmax,3),np.float32)
    for i,r in rec.iterrows():
        candidates=np.where(gids!=str(r.region_group_id))[0]; same=candidates[np.sign(lat[candidates])==np.sign(float(r.latitude_deg))]
        if len(same): candidates=same
        rng=np.random.default_rng(args.seed+1000003*i); j=int(rng.choice(candidates)); source=norm[i:i+1]
        warped=elastic_warp(source,args.warp_px,rng); residual=norm[j:j+1]-blur(norm[j:j+1],args.residual_sigma); residual=torch.roll(residual,(int(rng.integers(-4,5)),int(rng.integers(-4,5))),dims=(-2,-1)); residual-=residual.mean(dim=(-2,-1),keepdim=True)
        base=torch.clamp(warped+args.residual_alpha*residual,-1,1)
        for k,(deg,flip) in enumerate(transforms):
            syn=torch.clamp(rotate(base,deg)*flip,-1,1); pool[i][k]=syn; donor[i,k]=j
            generic[i,k]=float(torch.mean((syn-source).square()).item())
            z=smooth_cycle_descriptor(denormalize_gauss(syn),np.asarray([lat[i]]),np.asarray([times[i]]))[0]; desc[i,k]=((z-med)/scale).astype(np.float32)
    sel=np.argmin(generic,axis=1).astype(int); total,g,p=objective(sel,desc,generic,target,args.lambda_physics); trace=[{'pass':0,'objective':total,'generic_term':g,'population_physics_distance':p,'changes':0}]
    for ps in range(1,args.passes+1):
        changes=0
        for i in range(n):
            old=int(sel[i]); best=(total,old,g,p)
            for k in range(kmax):
                if k==old: continue
                sel[i]=k; o,gg,pp=objective(sel,desc,generic,target,args.lambda_physics)
                if o<best[0]-1e-12: best=(o,k,gg,pp)
            sel[i]=best[1]; changes+=int(best[1]!=old); total,g,p=best[0],best[2],best[3]
        trace.append({'pass':ps,'objective':total,'generic_term':g,'population_physics_distance':p,'changes':changes})
        if changes==0: break
    rows=[]; sels=[]
    tag=f'hjcycle_l{int(round(args.lambda_physics*100)):03d}'
    for i,r in rec.iterrows():
        k=int(sel[i]); j=int(donor[i,k]); deg,flip=transforms[k]; a=pool[i][k][0,0].numpy().astype(np.float32); sid=f'{tag}_{args.seed}_{r.region_group_id}_{i:06d}'; q=arrdir/f'{sid}.npy'; np.save(q,a)
        rows.append({'synthetic_id':sid,'array_path':str(q.resolve()),'source_region_group_id':str(r.region_group_id),'donor_region_group_id':str(rec.iloc[j].region_group_id),'source_sample_id':str(r.sample_id),'donor_sample_id':str(rec.iloc[j].sample_id),'source_t_rec':str(r.t_rec),'latitude_deg':float(r.latitude_deg),'label_m1plus_24h':1,'generator_condition':'morphology_hj_cycle','generator_seed':args.seed,'warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'lambda_physics':args.lambda_physics,'rotation_deg':deg,'polarity_multiplier':flip,'selected_candidate':k})
        sels.append({'synthetic_id':sid,'selected_candidate':k,'rotation_deg':deg,'polarity_multiplier':flip,'generic_change':float(generic[i,k]),'soft_descriptor':json.dumps(desc[i,k].astype(float).tolist())})
    pd.DataFrame(rows).to_csv(out/'synthetic_manifest.csv',index=False); pd.DataFrame(sels).to_csv(out/'selection_trace.csv',index=False); (out/'optimization_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    summary={'family':'cycle-aware train-only Hale/Joy morphology selector','lambda_physics':args.lambda_physics,'cycle25_start_utc':str(CYCLE25_START),'candidate_transforms':transforms,'final_soft_population_distance':float(p),'final_generic_term':float(g),'forecast_outcomes_accessed':False}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
