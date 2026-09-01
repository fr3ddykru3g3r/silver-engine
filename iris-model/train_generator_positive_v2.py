from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import build_records, cache_records, MagnetogramDataset
from generator import ConditionalUNet, Diffusion
from preprocess import denormalize_gauss, preprocess_fits
from physics_v2 import population_distribution_loss_v2, pil_distribution_loss_v2
from generic_fidelity import generic_distribution_loss
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


class PreparedDataset(Dataset):
    """Read preprocessed fields from fixed NumPy arrays instead of FITS per step."""

    def __init__(self, records: pd.DataFrame, x_array: np.ndarray, raw_array: np.ndarray):
        self.records = records.reset_index(drop=True)
        self.x_array = x_array
        self.raw_array = raw_array

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int):
        row = self.records.iloc[i]
        return {
            'x': torch.from_numpy(np.array(self.x_array[i], copy=True)).float(),
            'raw_gauss': torch.from_numpy(np.array(self.raw_array[i], copy=True)).float(),
            'y': torch.tensor(float(row.label_m1plus_24h), dtype=torch.float32),
            'latitude': torch.tensor(float(row.latitude_deg), dtype=torch.float32),
            'group': str(row.region_group_id),
            'sample_id': str(row.sample_id),
        }


def prepare_tensor_cache(records: pd.DataFrame, cache_dir: Path):
    """Materialize each FITS preprocessing result once for laptop-safe repeats."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(records), 1, 128, 128)
    metadata = {'sample_ids': [str(x) for x in records.sample_id], 'shape': list(shape), 'dtype': 'float32'}
    metadata_path = cache_dir / 'metadata.json'
    x_path = cache_dir / 'x.npy'
    raw_path = cache_dir / 'raw.npy'
    if metadata_path.is_file() and x_path.is_file() and raw_path.is_file():
        try:
            saved = json.loads(metadata_path.read_text())
            x_existing = np.load(x_path, mmap_mode='r')
            raw_existing = np.load(raw_path, mmap_mode='r')
            if saved == metadata and tuple(x_existing.shape) == shape and tuple(raw_existing.shape) == shape:
                print(f'using prepared tensor cache: {cache_dir}', flush=True)
                return x_existing, raw_existing
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    print(f'preparing fixed tensor cache: {len(records)} FITS files', flush=True)
    x_part = cache_dir / 'x.npy.part'
    raw_part = cache_dir / 'raw.npy.part'
    x_mem = np.lib.format.open_memmap(x_part, mode='w+', dtype=np.float32, shape=shape)
    raw_mem = np.lib.format.open_memmap(raw_part, mode='w+', dtype=np.float32, shape=shape)
    for i, row in enumerate(records.itertuples(index=False), 1):
        norm, raw = preprocess_fits(row.fits_path, float(row.CDELT1), float(row.CDELT2), float(row.RSUN_REF))
        x_mem[i - 1] = norm.numpy()
        raw_mem[i - 1] = raw.numpy()
        if i % 100 == 0 or i == len(records):
            print(f'prepared {i}/{len(records)}', flush=True)
    x_mem.flush(); raw_mem.flush()
    del x_mem, raw_mem
    x_part.replace(x_path); raw_part.replace(raw_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return np.load(x_path, mmap_mode='r'), np.load(raw_path, mmap_mode='r')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--condition',choices=['base','hj','pil','hj_pil'],required=True)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--per-group',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--physics-batch-size',type=int,default=16)
    ap.add_argument('--max-steps',type=int,default=1200); ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant')
    ap.add_argument('--lambda-generic',type=float,default=0.0,
                    help='Optional train-only generic descriptor stabilization during fine-tuning.')
    ap.add_argument('--lambda-hj',type=float,default=0.05); ap.add_argument('--lambda-pil',type=float,default=0.1)
    ap.add_argument('--physics-gradient-ratio',type=float,default=0.0,
                    help='Optional trust-region cap: weighted physics-gradient norm as a fraction of denoising-gradient norm.')
    ap.add_argument('--physics-warmup-steps',type=int,default=200); ap.add_argument('--diffusion-steps',type=int,default=100)
    ap.add_argument('--base-channels',type=int,default=16); ap.add_argument('--physics-max-t-frac',type=float,default=0.25)
    ap.add_argument('--download-workers',type=int,default=16); ap.add_argument('--generator-dropout',type=float,default=0.0)
    ap.add_argument('--ema-decay',type=float,default=0.999); ap.add_argument('--ema-warmup-steps',type=int,default=0)
    ap.add_argument('--init-checkpoint',default=None,help='Optional frozen BASE checkpoint. Loads weights before a physics-only fine-tune; architecture/diffusion settings must match.')
    ap.add_argument('--prepared-cache',default=None,help='Optional fixed tensor cache; avoids re-reading and preprocessing FITS at every update.')
    ap.add_argument('--threads',type=int,default=2,help='PyTorch intra-op CPU threads.')
    ap.add_argument('--checkpoint-every',type=int,default=100,help='Write an auditable model/history checkpoint every N steps.')
    args=ap.parse_args(); seed_all(args.seed)
    if args.lr <= 0 or args.lambda_generic < 0 or args.physics_gradient_ratio < 0 or args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError('Invalid learning rate, generic weight, physics gradient ratio, or EMA decay')
    if args.ema_warmup_steps < 0:
        raise ValueError('EMA warmup must be non-negative')
    if args.threads <= 0:
        raise ValueError('threads must be positive')
    if args.checkpoint_every <= 0:
        raise ValueError('checkpoint-every must be positive')

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Keep factorial conditions isolated even when a shared --out-dir is used.
    # The previous direct-write behavior could silently overwrite a completed
    # arm with the next condition.
    out_root=Path(args.out_dir)
    out=out_root / args.condition
    out.mkdir(parents=True,exist_ok=True)
    full=build_records(args.evidence_dir,'train')
    selected=positive_subset(full,args.per_group,args.seed)
    if selected.region_group_id.nunique()<20: raise RuntimeError('Too few positive physical regions')
    selected=cache_records(selected,Path(args.cache_dir),args.download_workers)
    tensor_cache = Path(args.prepared_cache) if args.prepared_cache else out / 'prepared_tensor_cache'
    x_array, raw_array = prepare_tensor_cache(selected, tensor_cache)
    ds=PreparedDataset(selected, x_array, raw_array)
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
    scheduler = None
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.max_steps), eta_min=args.lr * 0.05
        )
    mix_rng=make_rng(device,args.seed+100003); phys_rng=make_rng(device,args.seed+200003)
    phys_cut=max(0,int(round((args.diffusion_steps-1)*args.physics_max_t_frac)))
    piter=iter(phys_loader); history=[]; step=0

    def make_checkpoint(step_value: int):
        return {
            'model':model.state_dict(),'ema':ema.state_dict(),'condition':args.condition,
            'seed':args.seed,'steps':step_value,'diffusion_steps':args.diffusion_steps,
            'base_channels':args.base_channels,'lambda_generic':args.lambda_generic,
            'lambda_hj':args.lambda_hj,'lambda_pil':args.lambda_pil,'positive_only':True,
            'per_group':args.per_group,'ema_decay':args.ema_decay,
            'ema_warmup_steps':args.ema_warmup_steps,'lr_schedule':args.lr_schedule,
            'v2':True,'init_checkpoint':init_meta,
            'checkpoint_every':args.checkpoint_every,
        }

    def write_periodic_checkpoint(step_value: int) -> None:
        torch.save(make_checkpoint(step_value), out/f'generator_step_{step_value}.pt')
        (out/f'training_history_step_{step_value}.json').write_text(
            json.dumps(history,indent=2)+'\n'
        )

    while step<args.max_steps:
        for b in loader:
            x=b['x'].to(device); y=torch.ones(len(x),device=device); lat=b['lat'].to(device)
            t=torch.randint(0,args.diffusion_steps,(len(x),),device=device,generator=mix_rng)
            noise=randn_like_with(x,mix_rng); xt,_=diffusion.q_sample(x,t,noise=noise)
            eps=model(xt,t,y,lat); denoise=torch.mean((eps-noise).square())
            lgeneric=denoise*0.
            lhj=denoise*0.; lpil=denoise*0.
            if args.condition!='base' or args.lambda_generic > 0:
                try: pb=next(piter)
                except StopIteration: piter=iter(phys_loader); pb=next(piter)
                px=pb['x'].to(device); praw=pb['raw'].to(device); plat=pb['lat'].to(device); py=torch.ones(len(px),device=device)
                pt=torch.randint(0,phys_cut+1,(len(px),),device=device,generator=phys_rng)
                pnoise=randn_like_with(px,phys_rng); pxt,_=diffusion.q_sample(px,pt,noise=pnoise)
                peps=model(pxt,pt,py,plat); x0=torch.clamp(diffusion.x0_from_eps(pxt,pt,peps),-1,1); pfake=denormalize_gauss(x0)
                if args.lambda_generic > 0: lgeneric=generic_distribution_loss(pfake,praw)
                if args.condition in ('hj','hj_pil'): lhj=population_distribution_loss_v2(pfake,praw,plat)
                if args.condition in ('pil','hj_pil'): lpil=pil_distribution_loss_v2(pfake,praw,pixel_mm=2.0)
            step+=1; ramp=min(1.,step/max(1,args.physics_warmup_steps))
            physics_total=ramp*(args.lambda_generic*lgeneric+args.lambda_hj*lhj+args.lambda_pil*lpil)
            total=denoise+physics_total
            physics_gradient_scale=1.0
            params=tuple(p for p in model.parameters() if p.requires_grad)
            if args.physics_gradient_ratio > 0 and physics_total.requires_grad:
                # Both losses share the same forward graph. Retain it after
                # the first gradient query so the physics gradient can be
                # measured before the combined capped update is assembled.
                denoise_grads=torch.autograd.grad(denoise,params,allow_unused=True,retain_graph=True)
                physics_grads=torch.autograd.grad(physics_total,params,allow_unused=True)
                denoise_norm_sq=sum((g.detach().square().sum() for g in denoise_grads if g is not None), torch.zeros((),device=denoise.device))
                physics_norm_sq=sum((g.detach().square().sum() for g in physics_grads if g is not None), torch.zeros((),device=denoise.device))
                denoise_norm=torch.sqrt(denoise_norm_sq)
                physics_norm=torch.sqrt(physics_norm_sq)
                if float(physics_norm.detach()) > 0:
                    physics_gradient_scale=min(1.0, float((args.physics_gradient_ratio*denoise_norm/physics_norm).detach()))
                opt.zero_grad(set_to_none=True)
                for p,gd,gp in zip(params,denoise_grads,physics_grads):
                    p.grad=(torch.zeros_like(p) if gd is None else gd)+(torch.zeros_like(p) if gp is None else physics_gradient_scale*gp)
            else:
                opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if args.ema_warmup_steps and step <= args.ema_warmup_steps:
                ema.load_state_dict(model.state_dict())
            else:
                ema_update(ema,model,args.ema_decay)
            if scheduler is not None: scheduler.step()
            rec={'step':step,'loss':float(total.item()),'denoise':float(denoise.item()),'generic':float(lgeneric.item()),'hj':float(lhj.item()),'pil':float(lpil.item()),'ramp':float(ramp),'physics_gradient_scale':float(physics_gradient_scale),'learning_rate':float(opt.param_groups[0]['lr']),'ema_warmup_active':bool(args.ema_warmup_steps and step <= args.ema_warmup_steps)}; history.append(rec)
            if step%args.checkpoint_every==0:
                write_periodic_checkpoint(step)
            if step%100==0: print(json.dumps(rec),flush=True)
            if step>=args.max_steps: break

    ck=make_checkpoint(step); ck['final_checkpoint']=True
    torch.save(ck,out/'generator.pt')
    selected.to_csv(out/'training_subset.csv.gz',index=False,compression='gzip')
    (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary={'positive_only':True,'train_rows':len(selected),'train_groups':int(selected.region_group_id.nunique()),'steps':step,'condition':args.condition,'lambda_generic':args.lambda_generic,'lambda_hj':args.lambda_hj,'lambda_pil':args.lambda_pil,'physics_gradient_ratio':args.physics_gradient_ratio,'ema_decay':args.ema_decay,'ema_warmup_steps':args.ema_warmup_steps,'lr_schedule':args.lr_schedule,'device':str(device),'threads':args.threads,'prepared_cache':str(tensor_cache),'init_checkpoint':init_meta}
    (out/'run_config.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
