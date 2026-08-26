from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import build_records, deterministic_smoke_subset, cache_records, MagnetogramDataset
from generator import ConditionalUNet, Diffusion
from physics import population_loss, pil_distribution_loss
from preprocess import denormalize_gauss


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        'x':torch.stack([b['x'] for b in batch]),
        'raw':torch.stack([b['raw_gauss'] for b in batch]),
        'y':torch.stack([b['y'] for b in batch]),
        'lat':torch.stack([b['latitude'] for b in batch]),
    }


def temporal_even(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    z=df.sort_values('t_rec').reset_index(drop=True)
    if len(z) <= n:
        return z.copy()
    idx=np.unique(np.round(np.linspace(0,len(z)-1,n)).astype(int))
    if len(idx) < n:
        used=set(idx.tolist())
        idx=np.concatenate([idx,np.asarray([i for i in range(len(z)) if i not in used][:n-len(idx)])])
    return z.iloc[np.sort(idx[:n])].copy()


def group_temporal_subset(records: pd.DataFrame, per_group: int, positive_slots: int, seed: int) -> pd.DataFrame:
    """Keep every training connected-region group while suppressing hourly autocorrelation.

    Positive groups contribute temporally spread positive observations plus negative
    observations where available; non-flaring groups contribute temporally spread
    negatives. This reduces download/compute cost without turning long-lived regions
    into hundreds of pseudo-independent training examples.
    """
    if per_group <= 0:
        return records.copy().reset_index(drop=True)
    pieces=[]
    for gid,g in records.groupby('region_group_id',sort=True):
        pos=g[g.label_m1plus_24h.eq(1)]
        neg=g[g.label_m1plus_24h.eq(0)]
        kp=min(max(0,positive_slots),len(pos),per_group)
        kn=min(per_group-kp,len(neg))
        if kp == 0:
            kn=min(per_group,len(neg))
        z=pd.concat([temporal_even(pos,kp),temporal_even(neg,kn)],ignore_index=True)
        if len(z) < per_group:
            rest=g[~g.sample_id.isin(z.sample_id)]
            z=pd.concat([z,temporal_even(rest,min(per_group-len(z),len(rest)))],ignore_index=True)
        pieces.append(z)
    out=pd.concat(pieces,ignore_index=True)
    return out.sample(frac=1,random_state=seed).reset_index(drop=True)


def sampler_for(records: pd.DataFrame, target_positive_fraction: float, seed: int):
    """Class-targeted and connected-region-balanced replacement sampler."""
    y=records.label_m1plus_24h.astype(int).to_numpy()
    p=float(np.clip(target_positive_fraction,1e-3,1-1e-3))
    npos=max(1,int(y.sum())); nneg=max(1,int((1-y).sum()))
    class_ratio=(p/(1-p))*(nneg/npos)
    group_sizes=records.groupby('region_group_id').size().to_dict()
    inv_group=np.asarray([1.0/max(1,int(group_sizes[g])) for g in records.region_group_id],dtype=np.float64)
    class_weight=np.where(y==1,class_ratio,1.0).astype(np.float64)
    weights=class_weight*inv_group
    g=torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),num_samples=len(records),replacement=True,generator=g)


@torch.no_grad()
def ema_update(ema, model, decay: float):
    for e,p in zip(ema.parameters(),model.parameters()): e.data.mul_(decay).add_(p.data,alpha=1-decay)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',default='cache/generator')
    ap.add_argument('--out-dir',default='outputs/generator'); ap.add_argument('--condition',choices=['base','hj','pil','hj_pil'],required=True)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--max-steps',type=int,default=0)
    ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--lambda-hj',type=float,default=0.05); ap.add_argument('--lambda-pil',type=float,default=0.05)
    ap.add_argument('--diffusion-steps',type=int,default=400); ap.add_argument('--base-channels',type=int,default=48)
    ap.add_argument('--positive-fraction',type=float,default=0.40); ap.add_argument('--download-workers',type=int,default=12)
    ap.add_argument('--per-group',type=int,default=8,help='Deterministic temporal samples retained per connected training region; 0 keeps all rows.')
    ap.add_argument('--positive-slots',type=int,default=4,help='Maximum retained positive samples per positive connected region before filling with negatives.')
    ap.add_argument('--physics-max-t-frac',type=float,default=0.25,
                    help='Apply HJ/PIL losses only to positive samples at t <= this fraction of diffusion horizon; x0 estimates at very high noise are not physically interpretable.')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir)/args.condition; out.mkdir(parents=True,exist_ok=True)
    rec_full=build_records(args.evidence_dir,'train')
    rec=group_temporal_subset(rec_full,args.per_group,args.positive_slots,args.seed)
    if args.smoke:
        rec=deterministic_smoke_subset(rec,96,args.seed); args.epochs=1; args.max_steps=2; args.diffusion_steps=40; args.base_channels=16; args.batch_size=8
    subset_summary={
        'full_train_rows':len(rec_full),'selected_train_rows':len(rec),
        'full_groups':int(rec_full.region_group_id.nunique()),'selected_groups':int(rec.region_group_id.nunique()),
        'full_positive_rows':int(rec_full.label_m1plus_24h.sum()),'selected_positive_rows':int(rec.label_m1plus_24h.sum()),
        'full_positive_groups':int(rec_full.loc[rec_full.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
        'selected_positive_groups':int(rec.loc[rec.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
        'per_group':args.per_group,'positive_slots':args.positive_slots,'seed':args.seed,
    }
    (out/'training_subset_summary.json').write_text(json.dumps(subset_summary,indent=2)+'\n')
    rec.to_csv(out/'training_subset.csv.gz',index=False,compression='gzip')
    print(json.dumps(subset_summary),flush=True)
    rec=cache_records(rec,Path(args.cache_dir)/'train',args.download_workers)
    ds=MagnetogramDataset(rec); sampler=sampler_for(rec,args.positive_fraction,args.seed)
    loader=DataLoader(ds,batch_size=args.batch_size,sampler=sampler,num_workers=0,collate_fn=collate,drop_last=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ConditionalUNet(base=args.base_channels).to(device); ema=ConditionalUNet(base=args.base_channels).to(device); ema.load_state_dict(model.state_dict()); ema.eval()
    diffusion=Diffusion(args.diffusion_steps,device=device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    history=[]; step=0; phys_cut=max(0,int(round((args.diffusion_steps-1)*args.physics_max_t_frac)))
    for epoch in range(1,args.epochs+1):
        for b in loader:
            x=b['x'].to(device); raw=b['raw'].to(device); y=b['y'].to(device); lat=b['lat'].to(device)
            t=torch.randint(0,args.diffusion_steps,(len(x),),device=device); xt,noise=diffusion.q_sample(x,t)
            eps=model(xt,t,y,lat); denoise=torch.mean((eps-noise)**2); total=denoise
            x0hat=torch.clamp(diffusion.x0_from_eps(xt,t,eps),-1,1); fake_b=denormalize_gauss(x0hat)
            pos=(y>0.5) & (t<=phys_cut); lhj=x.sum()*0.0; lpil=x.sum()*0.0
            if int(pos.sum())>=2 and args.condition in ['hj','hj_pil']:
                lhj=population_loss(fake_b[pos],raw[pos],lat[pos]); total=total+args.lambda_hj*lhj
            if int(pos.sum())>=2 and args.condition in ['pil','hj_pil']:
                lpil=pil_distribution_loss(fake_b[pos],raw[pos],pixel_mm=256.0/128.0); total=total+args.lambda_pil*lpil
            opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ema_update(ema,model,0.999)
            step+=1; rec_h={'step':step,'epoch':epoch,'loss':float(total.item()),'denoise':float(denoise.item()),'hj':float(lhj.item()),'pil':float(lpil.item()),'positive_low_noise_in_batch':int(pos.sum()),'physics_t_cut':phys_cut}
            history.append(rec_h)
            if step%20==0 or args.smoke: print(json.dumps(rec_h),flush=True)
            if args.max_steps and step>=args.max_steps: break
        if args.max_steps and step>=args.max_steps: break
    ck={'model':model.state_dict(),'ema':ema.state_dict(),'condition':args.condition,'seed':args.seed,'steps':step,'diffusion_steps':args.diffusion_steps,'base_channels':args.base_channels,'lambda_hj':args.lambda_hj,'lambda_pil':args.lambda_pil,'positive_fraction':args.positive_fraction,'physics_max_t_frac':args.physics_max_t_frac,'per_group':args.per_group,'positive_slots':args.positive_slots}
    torch.save(ck,out/'generator.pt'); (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    (out/'run_config.json').write_text(json.dumps(vars(args)|{'device':str(device),'train_rows':len(rec),'steps_completed':step},indent=2)+'\n')
    print(json.dumps({'condition':args.condition,'device':str(device),'steps':step,'train_rows':len(rec),'checkpoint':str(out/'generator.pt')},indent=2),flush=True)

if __name__=='__main__': main()
