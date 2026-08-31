from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from data import build_records
from generator import ConditionalUNet, Diffusion


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def conditioning_groups(evidence_dir: str | Path, max_groups: int = 0, seed: int = 2026):
    x=build_records(evidence_dir,'train')
    x=x[x.label_m1plus_24h.eq(1)].copy()
    if x.empty: raise RuntimeError('No positive train rows')
    g=x.groupby('region_group_id').agg(latitude=('latitude_deg','median'),n_rows=('sample_id','size')).reset_index()
    g=g.sort_values('region_group_id').reset_index(drop=True)
    if max_groups and len(g)>max_groups:
        g=g.sample(n=max_groups,random_state=seed).sort_values('region_group_id').reset_index(drop=True)
    return g


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--evidence-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--per-group',type=int,default=8); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--sampling-steps',type=int,default=50)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--max-groups',type=int,default=0); ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base=int(ck.get('base_channels',48)); steps=int(ck.get('diffusion_steps',400)); condition=str(ck.get('condition','unknown'))
    model=ConditionalUNet(base=base).to(device); model.load_state_dict(ck.get('ema',ck['model'])); model.eval(); diffusion=Diffusion(steps,device=device)
    groups=conditioning_groups(args.evidence_dir,max_groups=(4 if args.smoke else args.max_groups),seed=args.seed)
    if args.smoke:
        args.per_group=min(args.per_group,2); args.sampling_steps=min(args.sampling_steps,10); args.batch_size=min(args.batch_size,4)
    rows=[]; serial=0
    for _,g in groups.iterrows():
        left=args.per_group
        while left>0:
            n=min(args.batch_size,left)
            label=torch.ones(n,device=device); lat=torch.full((n,),float(g.latitude),device=device)
            samples=diffusion.ddim_sample(model,n,label,lat,sampling_steps=args.sampling_steps,eta=0.0).cpu().numpy().astype(np.float32)
            for j in range(n):
                sid=f'{condition}_{args.seed}_{g.region_group_id}_{serial:06d}'; serial+=1
                path=arrdir/f'{sid}.npy'; np.save(path,samples[j,0])
                rows.append({'synthetic_id':sid,'array_path':str(path.resolve()),'source_region_group_id':g.region_group_id,'latitude_deg':float(g.latitude),'label_m1plus_24h':1,'generator_condition':condition,'generator_seed':args.seed,'sampling_steps':args.sampling_steps})
            left-=n
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False)
    summary={'condition':condition,'checkpoint':str(args.checkpoint),'device':str(device),'positive_source_groups':len(groups),'per_group':args.per_group,'synthetic_count':len(man),'sampling_steps':args.sampling_steps,'seed':args.seed}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
