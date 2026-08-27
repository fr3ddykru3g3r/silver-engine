from __future__ import annotations

"""Train-only population-level physics-selected morphology bootstrap generator.

A fixed morphology-bootstrap candidate pool is created for every positive training
magnetogram. Selection uses *soft population-distribution* descriptors from
physics_v2, while the scientific manipulation gate is evaluated later with the
independent hard-threshold evaluator in evaluate_generator_v2.py.

The earlier per-image selector pulled every candidate toward the population median.
That collapses descriptor variance and can make the *distribution* farther from
real even when each individual score improves. This implementation instead chooses
one candidate per source jointly so the selected population matches the real
positive descriptor distribution while retaining a source-preservation penalty.

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


def energy_distance_multivariate(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance for small standardized descriptor populations."""
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.ndim==1: a=a[:,None]
    if b.ndim==1: b=b[:,None]
    if len(a)<2 or len(b)<2: return float('inf')
    ab=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1).mean()
    aa=np.linalg.norm(a[:,None,:]-a[None,:,:],axis=-1).mean()
    bb=np.linalg.norm(b[:,None,:]-b[None,:,:],axis=-1).mean()
    return float(max(0.0,2.0*ab-aa-bb))


def descriptor_targets(raw: torch.Tensor, lat: np.ndarray, condition: str):
    lt=torch.from_numpy(lat.astype(np.float32))
    with torch.no_grad():
        if condition=='pil':
            real=strong_pil_gradient_descriptor_v2(raw,pixel_mm=2.0).cpu().numpy()
            med,scale=robust_center_scale(real)
            return {'all':((real-med)/scale).astype(np.float32),'med':med,'scale':scale}
        if condition=='hj':
            real=polarity_geometry_descriptor_v2(raw,lt).cpu().numpy()
            out={}
            for hemi,mask in [('north',lat>=0),('south',lat<0)]:
                rr=real[mask] if int(mask.sum())>=4 else real
                med,scale=robust_center_scale(rr)
                out[hemi]={'target':((rr-med)/scale).astype(np.float32),'med':med,'scale':scale}
            return out
    raise ValueError(condition)


def soft_descriptor(syn_norm: torch.Tensor, lat: float, condition: str, targets) -> np.ndarray:
    raw=denormalize_gauss(syn_norm)
    with torch.no_grad():
        if condition=='pil':
            z=strong_pil_gradient_descriptor_v2(raw,pixel_mm=2.0)[0].cpu().numpy()
            return ((z-targets['med'])/targets['scale']).astype(np.float32)
        z=polarity_geometry_descriptor_v2(raw,torch.tensor([lat],dtype=torch.float32))[0].cpu().numpy()
        t=targets['north' if lat>=0 else 'south']
        return ((z-t['med'])/t['scale']).astype(np.float32)


def population_objective(selected: np.ndarray, cand_desc: np.ndarray, generic: np.ndarray,
                         lat: np.ndarray, condition: str, targets, lam: float) -> tuple[float,float,float]:
    g=float(np.mean(generic[np.arange(len(selected)),selected]))
    if condition=='pil':
        z=cand_desc[np.arange(len(selected)),selected]
        phys=energy_distance_multivariate(z,targets['all'])
    else:
        vals=[]; weights=[]
        for hemi,mask in [('north',lat>=0),('south',lat<0)]:
            idx=np.where(mask)[0]
            if len(idx)<2: continue
            z=cand_desc[idx,selected[idx]]
            vals.append(energy_distance_multivariate(z,targets[hemi]['target']))
            weights.append(len(idx))
        phys=float(np.average(vals,weights=weights)) if vals else float('inf')
    return g+float(lam)*phys,g,phys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--condition',choices=['pil','hj'],required=True); ap.add_argument('--lambda-physics',type=float,required=True)
    ap.add_argument('--warp-px',type=float,default=2.0); ap.add_argument('--residual-alpha',type=float,default=0.15); ap.add_argument('--residual-sigma',type=float,default=4.0)
    ap.add_argument('--candidates',type=int,default=8); ap.add_argument('--passes',type=int,default=3)
    ap.add_argument('--per-group',type=int,default=2); ap.add_argument('--max-groups',type=int,default=64)
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
    targets=descriptor_targets(raw,lat,args.condition)

    n=len(rec); kmax=args.candidates
    syn_pool=[[None for _ in range(kmax)] for _ in range(n)]
    donor_pool=np.zeros((n,kmax),dtype=int)
    generic=np.zeros((n,kmax),dtype=np.float64)
    desc=[]
    for i,r in rec.iterrows():
        candidates=np.where(gids!=str(r.region_group_id))[0]
        same=candidates[np.sign(lat[candidates])==np.sign(float(r.latitude_deg))]
        if len(same): candidates=same
        if len(candidates)==0: raise RuntimeError('Need at least two positive region groups')
        source=norm[i:i+1]; row_desc=[]
        for k in range(kmax):
            rng=np.random.default_rng(args.seed + 1000003*i + 7919*k)
            j=int(rng.choice(candidates))
            warped=elastic_warp(source,args.warp_px,rng)
            donor=norm[j:j+1]; residual=donor-blur(donor,args.residual_sigma)
            dy=int(rng.integers(-4,5)); dx=int(rng.integers(-4,5)); residual=torch.roll(residual,(dy,dx),dims=(-2,-1))
            residual=residual-residual.mean(dim=(-2,-1),keepdim=True)
            syn=torch.clamp(warped+args.residual_alpha*residual,-1,1)
            syn_pool[i][k]=syn; donor_pool[i,k]=j
            generic[i,k]=float(torch.mean((syn-source).square()).item())
            row_desc.append(soft_descriptor(syn,float(r.latitude_deg),args.condition,targets))
        desc.append(row_desc)
    cand_desc=np.asarray(desc,dtype=np.float32)

    # Start from the most source-preserving candidate per row, then deterministic
    # coordinate descent on the *population* objective. Candidate pools are fixed
    # across lambda values, preserving the coefficient ablation.
    selected=np.argmin(generic,axis=1).astype(int)
    trace=[]
    total,gterm,pterm=population_objective(selected,cand_desc,generic,lat,args.condition,targets,args.lambda_physics)
    trace.append({'pass':0,'objective':total,'generic_term':gterm,'population_physics_distance':pterm,'changes':0})
    for ps in range(1,args.passes+1):
        changes=0
        for i in range(n):
            old=int(selected[i]); best=(total,old,gterm,pterm)
            for k in range(kmax):
                if k==old: continue
                selected[i]=k
                obj,gg,pp=population_objective(selected,cand_desc,generic,lat,args.condition,targets,args.lambda_physics)
                if obj < best[0]-1e-12:
                    best=(obj,k,gg,pp)
            selected[i]=best[1]
            if best[1]!=old: changes+=1
            total,gterm,pterm=best[0],best[2],best[3]
        trace.append({'pass':ps,'objective':total,'generic_term':gterm,'population_physics_distance':pterm,'changes':changes})
        if changes==0: break

    rows=[]; selection_rows=[]
    for i,r in rec.iterrows():
        k=int(selected[i]); j=int(donor_pool[i,k]); syn=syn_pool[i][k]
        a=syn[0,0].numpy().astype(np.float32)
        tag=f'{args.condition}_l{int(round(args.lambda_physics*100)):03d}'
        sid=f'morphphys_{tag}_{args.seed}_{r.region_group_id}_{i:06d}'
        p=arrdir/f'{sid}.npy'; np.save(p,a)
        rows.append({'synthetic_id':sid,'array_path':str(p.resolve()),'source_region_group_id':str(r.region_group_id),'donor_region_group_id':str(rec.iloc[j].region_group_id),'source_sample_id':str(r.sample_id),'donor_sample_id':str(rec.iloc[j].sample_id),'latitude_deg':float(r.latitude_deg),'label_m1plus_24h':1,'generator_condition':f'morphology_{args.condition}','generator_seed':args.seed,'warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'lambda_physics':args.lambda_physics,'selected_candidate':k})
        selection_rows.append({'synthetic_id':sid,'selected_candidate':k,'generic_change':float(generic[i,k]),'soft_descriptor':json.dumps(cand_desc[i,k].astype(float).tolist())})
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(out/'selection_trace.csv',index=False)
    (out/'optimization_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    summary={'family':'train-only population-level physics-selected morphology bootstrap','condition':args.condition,'lambda_physics':args.lambda_physics,'warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'candidate_pool':args.candidates,'coordinate_passes':args.passes,'source_rows':len(rec),'source_groups':int(rec.region_group_id.nunique()),'synthetic_count':len(man),'selection_metric':'population energy distance in soft physics_v2 descriptor space + mean source-preservation penalty','independent_gate_metric':'evaluate_generator_v2 hard descriptors','final_soft_population_distance':float(pterm),'final_generic_term':float(gterm),'forecast_outcomes_accessed':False}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
