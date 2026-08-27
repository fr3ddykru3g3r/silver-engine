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


def conditioning_groups(evidence_dir,max_groups=0,seed=2026):
    x=build_records(evidence_dir,'train'); x=x[x.label_m1plus_24h.eq(1)].copy()
    if x.empty: raise RuntimeError('No positive train rows')
    g=x.groupby('region_group_id').agg(latitude=('latitude_deg','median')).reset_index().sort_values('region_group_id')
    if max_groups and len(g)>max_groups: g=g.sample(n=max_groups,random_state=seed).sort_values('region_group_id')
    return g.reset_index(drop=True)


def v_to_eps(v,xt,abar):
    a=torch.sqrt(abar); s=torch.sqrt(1-abar)
    return a*v+s*xt

@torch.no_grad()
def ddim_v(diffusion,model,n,label,latitude,sampling_steps=100,shape=(1,128,128)):
    sampling_steps=max(2,min(int(sampling_steps),diffusion.steps))
    seq=torch.linspace(0,diffusion.steps-1,sampling_steps,device=diffusion.device).round().long().unique()
    x=torch.randn((n,*shape),device=diffusion.device)
    for j in reversed(range(len(seq))):
        i=int(seq[j].item()); t=torch.full((n,),i,device=diffusion.device,dtype=torch.long); abar=diffusion.ab[i]
        v=model(x,t,label,latitude); eps=v_to_eps(v,x,abar)
        x0=torch.clamp(torch.sqrt(abar)*x-torch.sqrt(1-abar)*v,-1,1)
        if j==0: x=x0; continue
        ip=int(seq[j-1].item()); abar_prev=diffusion.ab[ip]
        x=torch.sqrt(abar_prev)*x0+torch.sqrt(torch.clamp(1-abar_prev,min=0.0))*eps
    return torch.clamp(x,-1,1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--evidence-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--per-group',type=int,default=2); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--sampling-steps',type=int,default=100); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--max-groups',type=int,default=64)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu')
    if ck.get('prediction_type')!='v': raise RuntimeError('Checkpoint is not v-prediction')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=ConditionalUNet(base=int(ck.get('base_channels',16))).to(device); model.load_state_dict(ck.get('ema',ck['model'])); model.eval(); diffusion=Diffusion(int(ck.get('diffusion_steps',100)),device=device)
    groups=conditioning_groups(args.evidence_dir,args.max_groups,args.seed); rows=[]; serial=0
    for _,g in groups.iterrows():
        left=args.per_group
        while left>0:
            n=min(args.batch_size,left); label=torch.ones(n,device=device); lat=torch.full((n,),float(g.latitude),device=device)
            samples=ddim_v(diffusion,model,n,label,lat,args.sampling_steps).cpu().numpy().astype(np.float32)
            for j in range(n):
                sid=f'vpred_{args.seed}_{g.region_group_id}_{serial:06d}'; serial+=1; path=arrdir/f'{sid}.npy'; np.save(path,samples[j,0]); rows.append({'synthetic_id':sid,'array_path':str(path.resolve()),'source_region_group_id':g.region_group_id,'latitude_deg':float(g.latitude),'label_m1plus_24h':1,'generator_condition':'base_vpred','generator_seed':args.seed,'sampling_steps':args.sampling_steps})
            left-=n
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False); (out/'sampling_summary.json').write_text(json.dumps({'synthetic_count':len(man),'positive_source_groups':len(groups),'sampling_steps':args.sampling_steps,'prediction_type':'v'},indent=2)+'\n')

if __name__=='__main__': main()
