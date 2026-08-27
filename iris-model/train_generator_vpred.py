from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import build_records, cache_records, MagnetogramDataset
from generator import ConditionalUNet, Diffusion
from train_generator_v2 import collate, temporal_even, ema_update, make_rng, randn_like_with


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def positive_subset(df:pd.DataFrame, per_group:int, seed:int)->pd.DataFrame:
    x=df[df.label_m1plus_24h.eq(1)].copy(); parts=[]
    for _,g in x.groupby('region_group_id',sort=True):
        parts.append(temporal_even(g, per_group) if per_group>0 else g.copy())
    if not parts: raise RuntimeError('No positive training rows')
    return pd.concat(parts,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)


def region_balanced_sampler(df:pd.DataFrame, seed:int):
    sizes=df.groupby('region_group_id').size().to_dict()
    w=np.asarray([1.0/max(1,int(sizes[g])) for g in df.region_group_id],dtype=np.float64)
    gen=torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),num_samples=len(df),replacement=True,generator=gen)


def v_target(x0, noise, diffusion, t):
    a=diffusion.sqrt_ab[t][:,None,None,None]
    s=diffusion.sqrt_om[t][:,None,None,None]
    return a*noise-s*x0


def minsnr_v_weight(diffusion,t,gamma:float):
    ab=diffusion.ab[t]
    snr=ab/torch.clamp(1-ab,min=1e-8)
    return torch.minimum(snr,torch.full_like(snr,float(gamma)))/(snr+1.0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--per-group',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--max-steps',type=int,default=2000)
    ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--diffusion-steps',type=int,default=100)
    ap.add_argument('--base-channels',type=int,default=16); ap.add_argument('--min-snr-gamma',type=float,default=5.0)
    ap.add_argument('--download-workers',type=int,default=16); ap.add_argument('--generator-dropout',type=float,default=0.0)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    full=build_records(args.evidence_dir,'train'); selected=positive_subset(full,args.per_group,args.seed)
    if selected.region_group_id.nunique()<20: raise RuntimeError('Too few positive physical regions')
    selected=cache_records(selected,Path(args.cache_dir),args.download_workers)
    ds=MagnetogramDataset(selected)
    loader=DataLoader(ds,batch_size=args.batch_size,sampler=region_balanced_sampler(selected,args.seed),num_workers=0,collate_fn=collate,drop_last=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ConditionalUNet(base=args.base_channels,dropout=args.generator_dropout).to(device)
    ema=ConditionalUNet(base=args.base_channels,dropout=args.generator_dropout).to(device); ema.load_state_dict(model.state_dict()); ema.eval()
    diffusion=Diffusion(args.diffusion_steps,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    rng=make_rng(device,args.seed+100003); history=[]; step=0

    while step<args.max_steps:
        for b in loader:
            x=b['x'].to(device); y=torch.ones(len(x),device=device); lat=b['lat'].to(device)
            t=torch.randint(0,args.diffusion_steps,(len(x),),device=device,generator=rng)
            noise=randn_like_with(x,rng); xt,_=diffusion.q_sample(x,t,noise=noise)
            pred=model(xt,t,y,lat); target=v_target(x,noise,diffusion,t)
            per=((pred-target).square()).flatten(1).mean(1); w=minsnr_v_weight(diffusion,t,args.min_snr_gamma)
            loss=(w*per).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ema_update(ema,model,0.999)
            step+=1
            if step%100==0:
                rec={'step':step,'loss':float(loss.item()),'weight_mean':float(w.mean().item())}; history.append(rec); print(json.dumps(rec),flush=True)
            if step>=args.max_steps: break

    ck={'model':model.state_dict(),'ema':ema.state_dict(),'condition':'base','seed':args.seed,'steps':step,'diffusion_steps':args.diffusion_steps,'base_channels':args.base_channels,'prediction_type':'v','min_snr_gamma':args.min_snr_gamma,'positive_only':True,'per_group':args.per_group,'v2_objective_redesign':True}
    torch.save(ck,out/'generator.pt'); selected.to_csv(out/'training_subset.csv.gz',index=False,compression='gzip')
    (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    (out/'run_config.json').write_text(json.dumps(vars(args)|{'device':str(device),'steps_completed':step},indent=2)+'\n')
    print(json.dumps({'device':str(device),'steps':step,'prediction_type':'v','min_snr_gamma':args.min_snr_gamma},indent=2),flush=True)

if __name__=='__main__': main()
