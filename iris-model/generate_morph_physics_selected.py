from __future__ import annotations

"""Train-only physics-selected morphology bootstrap generator.

A fixed morphology-bootstrap candidate pool is created for each positive training
magnetogram. Candidate selection uses *soft* magnetic descriptors from physics_v2,
while the scientific manipulation gate is evaluated later with the independent
hard-threshold evaluator in evaluate_generator_v2.py.

No forecast validation/test outcomes are accessed here.
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data import build_records, cache_records, MagnetogramDataset
from preprocess import normalize_gauss, denormalize_gauss
from train_generator_v2 import group_temporal_subset
from generate_residual_bootstrap import blur
from generate_morph_bootstrap import elastic_warp
from physics_v2 import polarity_geometry_descriptor_v2, strong_pil_gradient_descriptor_v2


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def robust_center_scale(x: np.ndarray):
    med=np.nanmedian(x,axis=0)
    q1=np.nanpercentile(x,25,axis=0); q3=np.nanpercentile(x,75,axis=0)
    scale=np.where((q3-q1)>1e-6,q3-q1,np.nanstd(x,axis=0)+1e-6)
    scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.0)
    return med.astype(np.float32),scale.astype(np.float32)


def soft_targets(raw: torch.Tensor, lat: np.ndarray):
    lt=torch.from_numpy(lat.astype(np.float32))
    with torch.no_grad():
        pil=strong_pil_gradient_descriptor_v2(raw,pixel_mm=2.0).cpu().numpy()
        geo=polarity_geometry_descriptor_v2(raw,lt).cpu().numpy()
    pmed,pscale=robust_center_scale(pil)
    g={}
    for hemi,mask in [('north',lat>=0),('south',lat<0)]:
        if int(mask.sum())<4:
            g[hemi]=robust_center_scale(geo)
        else:
            g[hemi]=robust_center_scale(geo[mask])
    return pmed,pscale,g


def candidate_score(syn_norm: torch.Tensor, source_norm: torch.Tensor, lat: float,
                    condition: str, lam: float, pmed, pscale, gtargets):
    # Generic/source-preservation term competes with the physics term, making
    # lambda a genuine selection coefficient rather than a cosmetic multiplier.
    generic=float(torch.mean((syn_norm-source_norm).square()).item())
    raw=denormalize_gauss(syn_norm)
    lt=torch.tensor([lat],dtype=torch.float32)
    with torch.no_grad():
        if condition=='pil':
            z=strong_pil_gradient_descriptor_v2(raw,pixel_mm=2.0)[0].cpu().numpy()
            phys=float(np.mean(((z-pmed)/pscale)**2))
        elif condition=='hj':
            z=polarity_geometry_descriptor_v2(raw,lt)[0].cpu().numpy()
            med,scale=gtargets['north' if lat>=0 else 'south']
            phys=float(np.mean(((z-med)/scale)**2))
        else:
            raise ValueError(condition)
    return generic + float(lam)*phys, generic, phys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--condition',choices=['pil','hj'],required=True); ap.add_argument('--lambda-physics',type=float,required=True)
    ap.add_argument('--warp-px',type=float,default=2.0); ap.add_argument('--residual-alpha',type=float,default=0.15); ap.add_argument('--residual-sigma',type=float,default=4.0)
    ap.add_argument('--candidates',type=int,default=8); ap.add_argument('--per-group',type=int,default=2); ap.add_argument('--max-groups',type=int,default=64)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--download-workers',type=int,default=16)
    args=ap.parse_args(); seed_all(args.seed)
    if args.lambda_physics<=0: raise ValueError('lambda-physics must be >0')
    if args.candidates<2: raise ValueError('candidates must be >=2')

    out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    rec=build_records(args.evidence_dir,'train'); rec=rec[rec.label_m1plus_24h.eq(1)].copy()
    rec=group_temporal_subset(rec,args.per_group,args.per_group,args.seed)
    groups=sorted(rec.region_group_id.astype(str).unique())
    if args.max_groups and len(groups)>args.max_groups:
        rr=np.random.default_rng(args.seed); groups=sorted(rr.choice(groups,args.max_groups,replace=False).tolist()); rec=rec[rec.region_group_id.astype(str).isin(groups)].copy()
    rec=cache_records(rec,Path(args.cache_dir),args.download_workers).reset_index(drop=True)
    ds=MagnetogramDataset(rec)
    raw=torch.stack([ds[i]['raw_gauss'] for i in range(len(ds))]).float()
    norm=torch.from_numpy(np.stack([normalize_gauss(z[0].numpy()) for z in raw]))[:,None].float()
    lat=rec.latitude_deg.to_numpy(float); gids=rec.region_group_id.astype(str).to_numpy()
    pmed,pscale,gtargets=soft_targets(raw,lat)

    rows=[]; trace=[]
    for i,r in rec.iterrows():
        candidates=np.where(gids!=str(r.region_group_id))[0]
        same=candidates[np.sign(lat[candidates])==np.sign(float(r.latitude_deg))]
        if len(same): candidates=same
        if len(candidates)==0: raise RuntimeError('Need at least two positive region groups')
        source=norm[i:i+1]
        best=None
        for k in range(args.candidates):
            # Per-source/per-candidate RNG makes the candidate pool identical for
            # every lambda in a sweep and therefore supports a clean coefficient ablation.
            rng=np.random.default_rng(args.seed + 1000003*i + 7919*k)
            j=int(rng.choice(candidates))
            warped=elastic_warp(source,args.warp_px,rng)
            donor=norm[j:j+1]; residual=donor-blur(donor,args.residual_sigma)
            dy=int(rng.integers(-4,5)); dx=int(rng.integers(-4,5)); residual=torch.roll(residual,(dy,dx),dims=(-2,-1))
            residual=residual-residual.mean(dim=(-2,-1),keepdim=True)
            syn=torch.clamp(warped+args.residual_alpha*residual,-1,1)
            total,generic,phys=candidate_score(syn,source,float(r.latitude_deg),args.condition,args.lambda_physics,pmed,pscale,gtargets)
            item=(total,k,j,syn,generic,phys)
            if best is None or item[0]<best[0]: best=item
        total,k,j,syn,generic,phys=best
        a=syn[0,0].numpy().astype(np.float32)
        tag=f'{args.condition}_l{int(round(args.lambda_physics*100)):03d}'
        sid=f'morphphys_{tag}_{args.seed}_{r.region_group_id}_{i:06d}'
        p=arrdir/f'{sid}.npy'; np.save(p,a)
        rows.append({'synthetic_id':sid,'array_path':str(p.resolve()),'source_region_group_id':str(r.region_group_id),'donor_region_group_id':str(rec.iloc[j].region_group_id),'source_sample_id':str(r.sample_id),'donor_sample_id':str(rec.iloc[j].sample_id),'latitude_deg':float(r.latitude_deg),'label_m1plus_24h':1,'generator_condition':f'morphology_{args.condition}','generator_seed':args.seed,'warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'lambda_physics':args.lambda_physics,'selected_candidate':int(k)})
        trace.append({'synthetic_id':sid,'selected_candidate':int(k),'objective':float(total),'generic_change':float(generic),'soft_physics_score':float(phys)})
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False)
    pd.DataFrame(trace).to_csv(out/'selection_trace.csv',index=False)
    summary={'family':'train-only physics-selected morphology bootstrap','condition':args.condition,'lambda_physics':args.lambda_physics,'warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'candidate_pool':args.candidates,'source_rows':len(rec),'source_groups':int(rec.region_group_id.nunique()),'synthetic_count':len(man),'selection_metric':'soft physics_v2 descriptor + source-preservation competition','independent_gate_metric':'evaluate_generator_v2 hard descriptors','forecast_outcomes_accessed':False}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
