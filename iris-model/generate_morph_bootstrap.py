from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import build_records, cache_records, MagnetogramDataset
from preprocess import normalize_gauss
from train_generator_v2 import group_temporal_subset
from generate_residual_bootstrap import blur


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def smooth_displacement(h:int,w:int,warp_px:float,rng:np.random.Generator)->torch.Tensor:
    # Low-frequency elastic field: changes morphology while largely preserving
    # the field-strength distribution and local gradients. Train-only generator.
    coarse_h=max(3,h//32); coarse_w=max(3,w//32)
    d=torch.from_numpy(rng.normal(size=(1,2,coarse_h,coarse_w)).astype(np.float32))
    d=F.interpolate(d,size=(h,w),mode='bicubic',align_corners=True)
    # Remove rigid component and normalize RMS displacement.
    d=d-d.mean(dim=(-2,-1),keepdim=True)
    rms=torch.sqrt(torch.mean(d.square())).clamp_min(1e-6)
    return d*(float(warp_px)/float(rms))


def elastic_warp(x:torch.Tensor,warp_px:float,rng:np.random.Generator)->torch.Tensor:
    _,_,h,w=x.shape
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
    grid=torch.stack([xx,yy],dim=-1)[None]
    d=smooth_displacement(h,w,warp_px,rng)
    grid[...,0]+=2.0*d[:,0]/max(1,w-1)
    grid[...,1]+=2.0*d[:,1]/max(1,h-1)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='reflection',align_corners=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--warp-px',type=float,required=True); ap.add_argument('--residual-alpha',type=float,default=0.15); ap.add_argument('--residual-sigma',type=float,default=4.0)
    ap.add_argument('--per-group',type=int,default=2); ap.add_argument('--max-groups',type=int,default=64); ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--download-workers',type=int,default=16)
    args=ap.parse_args(); seed_all(args.seed)
    if not (0 < args.warp_px <= 16): raise ValueError('warp-px must be in (0,16]')
    if not (0 <= args.residual_alpha <= 0.3): raise ValueError('residual-alpha must be in [0,0.3]')

    out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    rec=build_records(args.evidence_dir,'train'); rec=rec[rec.label_m1plus_24h.eq(1)].copy()
    rec=group_temporal_subset(rec,args.per_group,args.per_group,args.seed)
    groups=sorted(rec.region_group_id.astype(str).unique())
    if args.max_groups and len(groups)>args.max_groups:
        rr=np.random.default_rng(args.seed); groups=sorted(rr.choice(groups,args.max_groups,replace=False).tolist()); rec=rec[rec.region_group_id.astype(str).isin(groups)].copy()
    rec=cache_records(rec,Path(args.cache_dir),args.download_workers)
    ds=MagnetogramDataset(rec)
    raw=torch.stack([ds[i]['raw_gauss'] for i in range(len(ds))]).float()
    norm=torch.from_numpy(np.stack([normalize_gauss(z[0].numpy()) for z in raw]))[:,None].float()

    rng=np.random.default_rng(args.seed); rows=[]
    gids=rec.region_group_id.astype(str).to_numpy(); lats=rec.latitude_deg.to_numpy(float)
    for i,r in rec.reset_index(drop=True).iterrows():
        candidates=np.where(gids!=str(r.region_group_id))[0]
        same=candidates[np.sign(lats[candidates])==np.sign(float(r.latitude_deg))]
        if len(same): candidates=same
        if len(candidates)==0: raise RuntimeError('Need at least two positive region groups')
        j=int(rng.choice(candidates))
        source=norm[i:i+1]
        warped=elastic_warp(source,args.warp_px,rng)
        donor=norm[j:j+1]
        residual=donor-blur(donor,args.residual_sigma)
        dy=int(rng.integers(-4,5)); dx=int(rng.integers(-4,5)); residual=torch.roll(residual,(dy,dx),dims=(-2,-1))
        residual=residual-residual.mean(dim=(-2,-1),keepdim=True)
        syn=torch.clamp(warped+args.residual_alpha*residual,-1,1)[0,0].numpy().astype(np.float32)
        sid=f'morph_w{int(round(args.warp_px)):02d}_a{int(round(args.residual_alpha*100)):02d}_{args.seed}_{r.region_group_id}_{i:06d}'
        p=arrdir/f'{sid}.npy'; np.save(p,syn)
        rows.append({'synthetic_id':sid,'array_path':str(p.resolve()),'source_region_group_id':str(r.region_group_id),'donor_region_group_id':str(rec.iloc[j].region_group_id),'source_sample_id':str(r.sample_id),'donor_sample_id':str(rec.iloc[j].sample_id),'latitude_deg':float(r.latitude_deg),'label_m1plus_24h':1,'generator_condition':'morphology_bootstrap','generator_seed':args.seed,'warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'residual_sigma':args.residual_sigma})
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False)
    summary={'family':'train-only morphology bootstrap','warp_px':args.warp_px,'residual_alpha':args.residual_alpha,'residual_sigma':args.residual_sigma,'seed':args.seed,'source_rows':len(rec),'source_groups':int(rec.region_group_id.nunique()),'synthetic_count':len(man),'forecast_outcomes_accessed':False}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
