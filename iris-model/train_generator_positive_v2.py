from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import build_records, cache_records, MagnetogramDataset
from generator import ConditionalUNet, Diffusion
from preprocess import denormalize_gauss
from physics_v2 import population_distribution_loss_v2, pil_distribution_loss_v2
from train_generator_v2 import collate, temporal_even, ema_update, make_rng, randn_like_with


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def positive_subset(df:pd.DataFrame, per_group:int, seed:int)->pd.DataFrame:
    x=df[df.label_m1plus_24h.eq(1)].copy()
    parts=[]
    for gid,g in x.groupby('region_group_id',sort=True):
        parts.append(temporal_even(g, per_group) if per_group>0 else g.copy())
    if not parts: raise RuntimeError('No positive training rows')
    out=pd.concat(parts,ignore_index=True)
    return out.sample(frac=1,random_state=seed).reset_index(drop=True)


def region_balanced_sampler(df:pd.DataFrame, seed:int):
    sizes=df.groupby('region_group_id').size().to_dict()
    w=np.asarray([1.0/max(1,int(sizes[g])) for g in df.region_group_id],dtype=np.float64)
    gen=torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double), num_samples=len(df), replacement=True, generator=gen)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--condition',choices=['base','hj','pil','hj_pil'],required=True)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--per-group',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--physics-batch-size',type=int,default=16)
    ap.add_argument('--max-steps',type=int,default=1200); ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--lambda-hj',type=float,default=0.05); ap.add_argument('--lambda-pil',type=float,default=0.1)
    ap.add_argument('--physics-warmup-steps',type=int,default=200); ap.add_argument('--diffusion-steps',type=int,default=100)
    ap.add_argument('--base-channels',type=int,default=16); ap.add_argument('--physics-max-t-frac',type=float,default=0.25)
    ap.add_argument('--download-workers',type=int,default=16); ap.add_argument('--generator-dropout',type=float,default=0.0)
    ap.add_argument('--init-checkpoint',default=None,help='Optional frozen BASE checkpoint. Loads weights before a physics-only fine-tune; architecture/diffusion settings must match.')
    args=ap.parse_args(); seed_all(args.seed)

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    full=build_records(args.evidence_dir,'train')
    selected=positive_subset(full,args.per_group,args.seed)
    if selected.region_group_id.nunique()<20: raise RuntimeError('Too few positive physical regions')
    selected=cache_records(selected,Path(args.cache_dir),args.download_workers)
    ds=MagnetogramDataset(selected)
    loader=DataLoader(ds,batch_size=args.batch_size,sampler=region_balanced_sampler(selected,args.seed),num_workers=0,collate_fn=collate,drop_last=True)
    phys_loader=DataLoader(ds,batch_size=args.physics_batch_size,sampler=region_balanced_sampler(selected,args.seed+991),num_workers=0,collate_fn=collate,drop_last=True)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ConditionalUNet(base=args.base_channels,dropout=args.generator_dropout).to(device)
    ema=ConditionalUNet(base=args.base_channels,dropout=args.generator_dropout).to(device)
    init_meta=None
    if args.init_checkpoint:
        ck0=torch.load(args.init_checkpoint,map_location='cpu')
        if int(ck0.get('base_channels',args.base_channels))!=args.base_channels: raise RuntimeError('init checkpoint base_channels mismatch')
        if int(ck0.get('diffusion_steps',args.diffusion_steps))!=args.diffusion_steps: raise RuntimeError('init checkpoint diffusion_steps mismatch')
        model.load_state_dict(ck0['ema'] if 'ema' in ck0 else ck0['model'])
        ema.load_state_dict(model.state_dict())
        init_meta={'path':str(args.init_checkpoint),'source_condition':ck0.get('condition'),'source_steps':ck0.get('steps'),'source_seed':ck0.get('seed')}
    else:
        ema.load_state_dict(model.state_dict())
    ema.eval()
    diffusion=Diffusion(args.diffusion_steps,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    mix_rng=make_rng(device,args.seed+100003); phys_rng=make_rng(device,args.seed+200003)
    phys_cut=max(0,int(round((args.diffusion_steps-1)*args.physics_max_t_frac)))
    piter=iter(phys_loader); history=[]; step=0

    while step<args.max_steps:
        for b in loader:
            x=b['x'].to(device); y=torch.ones(len(x),device=device); lat=b['lat'].to(device)
            t=torch.randint(0,args.diffusion_steps,(len(x),),device=device,generator=mix_rng)
            noise=randn_like_with(x,mix_rng); xt,_=diffusion.q_sample(x,t,noise=noise)
            eps=model(xt,t,y,lat); denoise=torch.mean((eps-noise).square())
            lhj=denoise*0.; lpil=denoise*0.
            if args.condition!='base':
                try: pb=next(piter)
                except StopIteration: piter=iter(phys_loader); pb=next(piter)
                px=pb['x'].to(device); praw=pb['raw'].to(device); plat=pb['lat'].to(device); py=torch.ones(len(px),device=device)
                pt=torch.randint(0,phys_cut+1,(len(px),),device=device,generator=phys_rng)
                pnoise=randn_like_with(px,phys_rng); pxt,_=diffusion.q_sample(px,pt,noise=pnoise)
                peps=model(pxt,pt,py,plat); x0=torch.clamp(diffusion.x0_from_eps(pxt,pt,peps),-1,1); pfake=denormalize_gauss(x0)
                if args.condition in ('hj','hj_pil'): lhj=population_distribution_loss_v2(pfake,praw,plat)
                if args.condition in ('pil','hj_pil'): lpil=pil_distribution_loss_v2(pfake,praw,pixel_mm=2.0)
            step+=1; ramp=min(1.,step/max(1,args.physics_warmup_steps))
            total=denoise+ramp*(args.lambda_hj*lhj+args.lambda_pil*lpil)
            opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ema_update(ema,model,0.999)
            rec={'step':step,'loss':float(total.item()),'denoise':float(denoise.item()),'hj':float(lhj.item()),'pil':float(lpil.item()),'ramp':float(ramp)}; history.append(rec)
            if step%100==0: print(json.dumps(rec),flush=True)
            if step>=args.max_steps: break

    ck={'model':model.state_dict(),'ema':ema.state_dict(),'condition':args.condition,'seed':args.seed,'steps':step,'diffusion_steps':args.diffusion_steps,'base_channels':args.base_channels,'lambda_hj':args.lambda_hj,'lambda_pil':args.lambda_pil,'positive_only':True,'per_group':args.per_group,'v2':True,'init_checkpoint':init_meta}
    torch.save(ck,out/'generator.pt')
    selected.to_csv(out/'training_subset.csv.gz',index=False,compression='gzip')
    (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary={'positive_only':True,'train_rows':len(selected),'train_groups':int(selected.region_group_id.nunique()),'steps':step,'condition':args.condition,'lambda_hj':args.lambda_hj,'lambda_pil':args.lambda_pil,'device':str(device),'init_checkpoint':init_meta}
    (out/'run_config.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
