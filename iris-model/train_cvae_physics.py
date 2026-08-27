from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import build_records, cache_records, MagnetogramDataset
from preprocess import denormalize_gauss
from cvae import MagnetogramCVAE, kl_standard_normal
from physics_v2 import population_distribution_loss_v2, pil_distribution_loss_v2


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def temporal_even(g: pd.DataFrame, n: int) -> pd.DataFrame:
    g = g.sort_values('t_rec').reset_index(drop=True)
    if n <= 0 or len(g) <= n: return g.copy()
    idx = np.unique(np.rint(np.linspace(0, len(g)-1, n)).astype(int))
    return g.iloc[idx].copy()


def positive_subset(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    x = df[df.label_m1plus_24h.eq(1)].copy()
    parts = [temporal_even(g, per_group) for _, g in x.groupby('region_group_id', sort=True)]
    if not parts: raise RuntimeError('No positive training rows')
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def region_balanced_sampler(df: pd.DataFrame, seed: int, draws: int = 0):
    sizes = df.groupby('region_group_id').size().to_dict()
    w = np.asarray([1.0 / max(1, int(sizes[g])) for g in df.region_group_id], np.float64)
    gen = torch.Generator().manual_seed(seed)
    n = int(draws) if draws > 0 else len(df)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), n, replacement=True, generator=gen)


def image_gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return gx, gy


def reconstruction_losses(recon: torch.Tensor, target: torch.Tensor, raw: torch.Tensor):
    # Baseline normalized-space fidelity.
    l1 = F.l1_loss(recon, target)
    # Give magnetically active pixels more influence without discarding the quiet field.
    strong = torch.sigmoid((raw.abs() - 150.0) / 50.0)
    pred_raw = denormalize_gauss(recon)
    raw_err = (pred_raw - raw).abs() / 500.0
    strong_l1 = (raw_err * (1.0 + 3.0 * strong)).mean()
    rgx, rgy = image_gradient(recon); tgx, tgy = image_gradient(target)
    grad = F.l1_loss(rgx, tgx) + F.l1_loss(rgy, tgy)
    return l1, strong_l1, grad, pred_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True); ap.add_argument('--cache-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--condition', choices=['base','pil','hj','hj_pil'], default='base')
    ap.add_argument('--seed', type=int, default=2026); ap.add_argument('--per-group', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=16); ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--base', type=int, default=16); ap.add_argument('--latent-dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--download-workers', type=int, default=12)
    ap.add_argument('--beta-kl', type=float, default=2e-4); ap.add_argument('--lambda-strong', type=float, default=0.20)
    ap.add_argument('--lambda-grad', type=float, default=0.10); ap.add_argument('--lambda-pil', type=float, default=0.0)
    ap.add_argument('--lambda-hj', type=float, default=0.0); ap.add_argument('--physics-warmup', type=int, default=200)
    ap.add_argument('--init-checkpoint', default='')
    args = ap.parse_args(); seed_all(args.seed)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rec = positive_subset(build_records(args.evidence_dir, 'train'), args.per_group, args.seed)
    if rec.region_group_id.nunique() < 20: raise RuntimeError('Too few positive physical regions')
    rec = cache_records(rec, Path(args.cache_dir), workers=args.download_workers)
    ds = MagnetogramDataset(rec)
    sampler = region_balanced_sampler(rec, args.seed, draws=max(len(rec), args.batch_size * 64))
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MagnetogramCVAE(args.base, args.latent_dim).to(device)
    if args.init_checkpoint:
        ck0 = torch.load(args.init_checkpoint, map_location='cpu')
        if int(ck0['base']) != args.base or int(ck0['latent_dim']) != args.latent_dim:
            raise RuntimeError('CVAE initialization architecture mismatch')
        model.load_state_dict(ck0['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1919)
    hist=[]; step=0

    while step < args.steps:
        for b in dl:
            x=b['x'].to(device); raw=b['raw_gauss'].to(device); lat=b['latitude'].to(device)
            recon,mu,logvar=model(x,lat,generator=rng)
            l1,lstrong,lgrad,pred_raw=reconstruction_losses(recon,x,raw)
            kl=kl_standard_normal(mu,logvar)
            lpil=x.sum()*0.; lhj=x.sum()*0.
            if args.condition in ('pil','hj_pil'):
                lpil=pil_distribution_loss_v2(pred_raw,raw,pixel_mm=2.0)
            if args.condition in ('hj','hj_pil'):
                lhj=population_distribution_loss_v2(pred_raw,raw,lat)
            step += 1
            ramp=min(1.0,step/max(1,args.physics_warmup))
            loss=l1 + args.lambda_strong*lstrong + args.lambda_grad*lgrad + args.beta_kl*kl + ramp*(args.lambda_pil*lpil + args.lambda_hj*lhj)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            r={'step':step,'loss':float(loss.item()),'l1':float(l1.item()),'strong':float(lstrong.item()),'grad':float(lgrad.item()),'kl':float(kl.item()),'pil':float(lpil.item()),'hj':float(lhj.item()),'ramp':ramp}
            hist.append(r)
            if step % 50 == 0: print(json.dumps(r),flush=True)
            if step >= args.steps: break

    ck={'model':model.state_dict(),'base':args.base,'latent_dim':args.latent_dim,'condition':args.condition,'seed':args.seed,'steps':args.steps,'beta_kl':args.beta_kl,'lambda_strong':args.lambda_strong,'lambda_grad':args.lambda_grad,'lambda_pil':args.lambda_pil,'lambda_hj':args.lambda_hj,'positive_only':True,'generator_family':'cvae_v3'}
    torch.save(ck,out/'cvae.pt')
    rec.drop(columns=['fits_path'],errors='ignore').to_csv(out/'training_subset.csv.gz',index=False,compression='gzip')
    (out/'training_history.json').write_text(json.dumps(hist,indent=2)+'\n')
    summary={'generator_family':'cvae_v3','condition':args.condition,'train_rows':len(rec),'train_groups':int(rec.region_group_id.nunique()),'steps':args.steps,'base':args.base,'latent_dim':args.latent_dim,'device':str(device),'init_checkpoint':args.init_checkpoint or None}
    (out/'run_config.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__': main()
