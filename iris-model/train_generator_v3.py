from __future__ import annotations

import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler

from data import build_records,cache_records,MagnetogramDataset
from generator import ConditionalUNet,Diffusion
from preprocess import denormalize_gauss
from physics_v2 import population_distribution_loss_v2,pil_distribution_loss_v2
from generic_fidelity import generic_distribution_loss


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def collate(batch):
    return {'x':torch.stack([b['x'] for b in batch]),'raw':torch.stack([b['raw_gauss'] for b in batch]),'y':torch.stack([b['y'] for b in batch]),'lat':torch.stack([b['latitude'] for b in batch])}

def temporal_even(df,n):
    z=df.sort_values('t_rec').reset_index(drop=True)
    if len(z)<=n:return z.copy()
    idx=np.unique(np.round(np.linspace(0,len(z)-1,n)).astype(int));return z.iloc[idx[:n]].copy()

def positive_subset(full:pd.DataFrame,per_group:int):
    p=full[full.label_m1plus_24h.eq(1)].copy();parts=[]
    for _,g in p.groupby('region_group_id',sort=True):parts.append(temporal_even(g,per_group))
    return pd.concat(parts,ignore_index=True)

def region_sampler(df,seed):
    sizes=df.groupby('region_group_id').size().to_dict();w=np.asarray([1/max(1,sizes[g]) for g in df.region_group_id],dtype=np.float64)
    return WeightedRandomSampler(torch.tensor(w,dtype=torch.double),len(df),replacement=True,generator=torch.Generator().manual_seed(seed))

@torch.no_grad()
def ema_update(ema,model,d=.999):
    for e,p in zip(ema.parameters(),model.parameters()):e.data.mul_(d).add_(p.data,alpha=1-d)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--cache-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--condition',choices=['base','hj','pil','hj_pil'],required=True);ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--per-group',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--max-steps',type=int,default=1200);ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--diffusion-steps',type=int,default=200);ap.add_argument('--base-channels',type=int,default=32);ap.add_argument('--lambda-generic',type=float,default=.08)
    ap.add_argument('--lambda-hj',type=float,default=.10);ap.add_argument('--lambda-pil',type=float,default=.10);ap.add_argument('--warmup-steps',type=int,default=150)
    ap.add_argument('--physics-max-t-frac',type=float,default=.20);ap.add_argument('--download-workers',type=int,default=16);a=ap.parse_args();seed_all(a.seed)
    out=Path(a.out_dir)/a.condition;out.mkdir(parents=True,exist_ok=True)
    full=build_records(a.evidence_dir,'train');rec=positive_subset(full,a.per_group)
    rec=cache_records(rec,Path(a.cache_dir),a.download_workers);ds=MagnetogramDataset(rec)
    dl=DataLoader(ds,batch_size=a.batch_size,sampler=region_sampler(rec,a.seed),num_workers=0,collate_fn=collate,drop_last=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=ConditionalUNet(base=a.base_channels).to(dev);ema=ConditionalUNet(base=a.base_channels).to(dev);ema.load_state_dict(model.state_dict());ema.eval();diff=Diffusion(a.diffusion_steps,dev);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    cut=max(1,int((a.diffusion_steps-1)*a.physics_max_t_frac));hist=[];step=0
    while step<a.max_steps:
      for b in dl:
        x=b['x'].to(dev);raw=b['raw'].to(dev);y=b['y'].to(dev);lat=b['lat'].to(dev)
        t=torch.randint(0,a.diffusion_steps,(len(x),),device=dev);xt,noise=diff.q_sample(x,t);eps=model(xt,t,y,lat);den=(eps-noise).square().mean()
        # Separate low-noise view of the SAME positive batch ensures every update carries the common realism objective.
        pt=torch.randint(0,cut+1,(len(x),),device=dev);pxt,pnoise=diff.q_sample(x,pt);peps=model(pxt,pt,y,lat);x0=torch.clamp(diff.x0_from_eps(pxt,pt,peps),-1,1);fake=denormalize_gauss(x0)
        lg=generic_distribution_loss(fake,raw);lhj=den*0;lpil=den*0
        if a.condition in ('hj','hj_pil'):lhj=population_distribution_loss_v2(fake,raw,lat)
        if a.condition in ('pil','hj_pil'):lpil=pil_distribution_loss_v2(fake,raw,2.0)
        step+=1;r=min(1.,step/max(1,a.warmup_steps));loss=den+r*(a.lambda_generic*lg+a.lambda_hj*lhj+a.lambda_pil*lpil)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();ema_update(ema,model)
        if step%50==0:print(json.dumps({'step':step,'loss':float(loss),'denoise':float(den),'generic':float(lg),'hj':float(lhj),'pil':float(lpil),'ramp':r}),flush=True)
        hist.append({'step':step,'loss':float(loss),'denoise':float(den),'generic':float(lg),'hj':float(lhj),'pil':float(lpil),'ramp':r})
        if step>=a.max_steps:break
    torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'condition':a.condition,'seed':a.seed,'steps':step,'diffusion_steps':a.diffusion_steps,'base_channels':a.base_channels,'v3':True,'lambda_generic':a.lambda_generic,'lambda_hj':a.lambda_hj,'lambda_pil':a.lambda_pil},out/'generator.pt')
    (out/'training_history.json').write_text(json.dumps(hist,indent=2)+'\n');(out/'run_config.json').write_text(json.dumps(vars(a)|{'positive_rows':len(rec),'positive_groups':int(rec.region_group_id.nunique()),'device':str(dev)},indent=2)+'\n')
if __name__=='__main__':main()
