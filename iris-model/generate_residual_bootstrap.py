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


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def gaussian_kernel1d(sigma:float, device='cpu'):
    radius=max(1,int(round(3*sigma)))
    x=torch.arange(-radius,radius+1,dtype=torch.float32,device=device)
    k=torch.exp(-0.5*(x/sigma)**2); return k/k.sum()


def blur(x:torch.Tensor,sigma:float)->torch.Tensor:
    k=gaussian_kernel1d(sigma,x.device); r=(len(k)-1)//2
    z=F.pad(x,(r,r,0,0),mode='reflect'); z=F.conv2d(z,k.view(1,1,1,-1))
    z=F.pad(z,(0,0,r,r),mode='reflect'); z=F.conv2d(z,k.view(1,1,-1,1))
    return z


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--alpha',type=float,required=True); ap.add_argument('--sigma',type=float,default=4.0)
    ap.add_argument('--per-group',type=int,default=2); ap.add_argument('--max-groups',type=int,default=64); ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--download-workers',type=int,default=16)
    args=ap.parse_args(); seed_all(args.seed)
    if not (0 < args.alpha <= 0.5): raise ValueError('alpha must be in (0,0.5]')

    out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    rec=build_records(args.evidence_dir,'train'); rec=rec[rec.label_m1plus_24h.eq(1)].copy()
    rec=group_temporal_subset(rec,args.per_group,args.per_group,args.seed)
    groups=sorted(rec.region_group_id.astype(str).unique())
    if args.max_groups and len(groups)>args.max_groups:
        rng=np.random.default_rng(args.seed); groups=sorted(rng.choice(groups,args.max_groups,replace=False).tolist()); rec=rec[rec.region_group_id.astype(str).isin(groups)].copy()
    rec=cache_records(rec,Path(args.cache_dir),args.download_workers)
    ds=MagnetogramDataset(rec)
    raws=[]
    for i in range(len(ds)): raws.append(ds[i]['raw_gauss'])
    raw=torch.stack(raws).float()  # N,1,H,W
    norm=torch.from_numpy(np.stack([normalize_gauss(z[0].numpy()) for z in raw]))[:,None].float()

    rng=np.random.default_rng(args.seed)
    rows=[]
    for i,r in rec.reset_index(drop=True).iterrows():
        candidates=np.where(rec.region_group_id.astype(str).to_numpy()!=str(r.region_group_id))[0]
        if len(candidates)==0: raise RuntimeError('Need at least two positive region groups')
        # Match donor hemisphere to avoid injecting population-inconsistent polarity geometry.
        same=candidates[np.sign(rec.iloc[candidates].latitude_deg.to_numpy())==np.sign(float(r.latitude_deg))]
        if len(same): candidates=same
        j=int(rng.choice(candidates))
        donor=norm[j:j+1]
        residual=donor-blur(donor,args.sigma)
        dy=int(rng.integers(-4,5)); dx=int(rng.integers(-4,5)); residual=torch.roll(residual,(dy,dx),dims=(-2,-1))
        # Zero-mean residual bootstrap preserves source large-scale polarity geometry while adding novel fine structure.
        residual=residual-residual.mean(dim=(-2,-1),keepdim=True)
        syn=torch.clamp(norm[i:i+1]+args.alpha*residual,-1,1)[0,0].numpy().astype(np.float32)
        sid=f'resboot_a{int(round(args.alpha*1000)):03d}_{args.seed}_{r.region_group_id}_{i:06d}'
        p=arrdir/f'{sid}.npy'; np.save(p,syn)
        rows.append({'synthetic_id':sid,'array_path':str(p.resolve()),'source_region_group_id':str(r.region_group_id),'donor_region_group_id':str(rec.iloc[j].region_group_id),'source_sample_id':str(r.sample_id),'donor_sample_id':str(rec.iloc[j].sample_id),'latitude_deg':float(r.latitude_deg),'label_m1plus_24h':1,'generator_condition':'residual_bootstrap','generator_seed':args.seed,'alpha':args.alpha,'sigma':args.sigma,'residual_shift_y':dy,'residual_shift_x':dx})
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False)
    summary={'family':'train-only residual bootstrap','alpha':args.alpha,'sigma':args.sigma,'seed':args.seed,'source_rows':len(rec),'source_groups':int(rec.region_group_id.nunique()),'synthetic_count':len(man),'forecast_outcomes_accessed':False}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
