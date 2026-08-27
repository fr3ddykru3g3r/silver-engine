from __future__ import annotations

import argparse
import json
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import build_records, cache_records, MagnetogramDataset
from generator import ConditionalUNet, Diffusion
from preprocess import denormalize_gauss
from physics_v2 import population_distribution_loss_v2, pil_distribution_loss_v2


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {
        'x': torch.stack([b['x'] for b in batch]),
        'raw': torch.stack([b['raw_gauss'] for b in batch]),
        'y': torch.stack([b['y'] for b in batch]),
        'lat': torch.stack([b['latitude'] for b in batch]),
    }


def temporal_even(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    z = df.sort_values('t_rec').reset_index(drop=True)
    if len(z) <= n:
        return z.copy()
    idx = np.unique(np.round(np.linspace(0, len(z) - 1, n)).astype(int))
    if len(idx) < n:
        used = set(idx.tolist())
        idx = np.concatenate([idx, np.asarray([i for i in range(len(z)) if i not in used][:n-len(idx)])])
    return z.iloc[np.sort(idx[:n])].copy()


def group_temporal_subset(records: pd.DataFrame, per_group: int, positive_slots: int, seed: int) -> pd.DataFrame:
    if per_group <= 0:
        return records.copy().reset_index(drop=True)
    pieces = []
    for _, g in records.groupby('region_group_id', sort=True):
        pos = g[g.label_m1plus_24h.eq(1)]
        neg = g[g.label_m1plus_24h.eq(0)]
        kp = min(max(0, positive_slots), len(pos), per_group)
        kn = min(per_group - kp, len(neg))
        if kp == 0:
            kn = min(per_group, len(neg))
        z = pd.concat([temporal_even(pos, kp), temporal_even(neg, kn)], ignore_index=True)
        if len(z) < per_group:
            rest = g[~g.sample_id.isin(z.sample_id)]
            z = pd.concat([z, temporal_even(rest, min(per_group-len(z), len(rest)))], ignore_index=True)
        pieces.append(z)
    out = pd.concat(pieces, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def mixed_sampler(records: pd.DataFrame, target_positive_fraction: float, seed: int):
    y = records.label_m1plus_24h.astype(int).to_numpy()
    p = float(np.clip(target_positive_fraction, 1e-3, 1-1e-3))
    npos = max(1, int(y.sum())); nneg = max(1, int((1-y).sum()))
    class_ratio = (p/(1-p)) * (nneg/npos)
    sizes = records.groupby('region_group_id').size().to_dict()
    inv_group = np.asarray([1.0/max(1, int(sizes[g])) for g in records.region_group_id], dtype=np.float64)
    weights = np.where(y == 1, class_ratio, 1.0).astype(np.float64) * inv_group
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(records), True, generator=gen)


def positive_group_sampler(records: pd.DataFrame, seed: int):
    sizes = records.groupby('region_group_id').size().to_dict()
    weights = np.asarray([1.0/max(1, int(sizes[g])) for g in records.region_group_id], dtype=np.float64)
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(records), True, generator=gen)


@torch.no_grad()
def ema_update(ema, model, decay: float):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.data.mul_(decay).add_(p.data, alpha=1-decay)


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def make_rng(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def randn_like_with(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--cache-dir', default='cache/generator_v2')
    ap.add_argument('--out-dir', default='outputs/generator_v2')
    ap.add_argument('--condition', choices=['base', 'hj', 'pil', 'hj_pil'], required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--physics-batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lambda-hj', type=float, default=0.05)
    ap.add_argument('--lambda-pil', type=float, default=0.05)
    ap.add_argument('--physics-warmup-steps', type=int, default=200)
    ap.add_argument('--diffusion-steps', type=int, default=400)
    ap.add_argument('--base-channels', type=int, default=48)
    ap.add_argument('--positive-fraction', type=float, default=0.40)
    ap.add_argument('--download-workers', type=int, default=12)
    ap.add_argument('--per-group', type=int, default=8)
    ap.add_argument('--positive-slots', type=int, default=4)
    ap.add_argument('--physics-max-t-frac', type=float, default=0.25)
    ap.add_argument('--generator-dropout', type=float, default=0.0,
                    help='Frozen at 0 for matched v2 physics ablations; avoids stochastic dropout-stream confounding.')
    args = ap.parse_args()

    seed_all(args.seed)
    out = Path(args.out_dir) / args.condition
    out.mkdir(parents=True, exist_ok=True)

    full = build_records(args.evidence_dir, 'train')
    selected = group_temporal_subset(full, args.per_group, args.positive_slots, args.seed)
    positives = selected[selected.label_m1plus_24h.eq(1)].copy().reset_index(drop=True)
    if positives.region_group_id.nunique() < 3:
        raise RuntimeError('Not enough positive physical regions for v2 physics training')

    summary = {
        'v2': True,
        'condition': args.condition,
        'full_rows': len(full),
        'selected_rows': len(selected),
        'selected_groups': int(selected.region_group_id.nunique()),
        'positive_rows': len(positives),
        'positive_groups': int(positives.region_group_id.nunique()),
        'always_on_positive_physics_batch': True,
        'matched_mixed_diffusion_rng_across_conditions': True,
        'generator_dropout': args.generator_dropout,
        'seed': args.seed,
    }
    (out/'training_subset_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    selected.to_csv(out/'training_subset.csv.gz', index=False, compression='gzip')

    selected = cache_records(selected, Path(args.cache_dir)/'mixed', args.download_workers)
    positives = selected[selected.label_m1plus_24h.eq(1)].copy().reset_index(drop=True)

    mixed_ds = MagnetogramDataset(selected)
    positive_ds = MagnetogramDataset(positives)
    mixed_loader = DataLoader(
        mixed_ds,
        batch_size=args.batch_size,
        sampler=mixed_sampler(selected, args.positive_fraction, args.seed),
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    physics_loader = DataLoader(
        positive_ds,
        batch_size=args.physics_batch_size,
        sampler=positive_group_sampler(positives, args.seed+991),
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalUNet(base=args.base_channels, dropout=args.generator_dropout).to(device)
    ema = ConditionalUNet(base=args.base_channels, dropout=args.generator_dropout).to(device)
    ema.load_state_dict(model.state_dict()); ema.eval()
    diffusion = Diffusion(args.diffusion_steps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    mix_rng = make_rng(device, args.seed + 100_003)
    phys_rng = make_rng(device, args.seed + 200_003)
    phys_cut = max(0, int(round((args.diffusion_steps-1)*args.physics_max_t_frac)))
    physics_iter = iter(physics_loader)
    history = []
    step = 0

    steps_per_epoch = max(1, len(mixed_loader))
    effective_epochs = args.epochs
    if args.max_steps > 0:
        effective_epochs = max(args.epochs, int(math.ceil(args.max_steps / steps_per_epoch)))

    for epoch in range(1, effective_epochs+1):
        for batch in mixed_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            lat = batch['lat'].to(device)
            t = torch.randint(0, args.diffusion_steps, (len(x),), device=device, generator=mix_rng)
            mixed_noise = randn_like_with(x, mix_rng)
            xt, noise = diffusion.q_sample(x, t, noise=mixed_noise)
            eps = model(xt, t, y, lat)
            denoise = torch.mean((eps-noise).square())

            lhj = denoise * 0.0
            lpil = denoise * 0.0
            if args.condition != 'base':
                pb, physics_iter = next_or_restart(physics_iter, physics_loader)
                px = pb['x'].to(device)
                praw = pb['raw'].to(device)
                py = pb['y'].to(device)
                plat = pb['lat'].to(device)
                pt = torch.randint(0, phys_cut+1, (len(px),), device=device, generator=phys_rng)
                physics_noise = randn_like_with(px, phys_rng)
                pxt, _ = diffusion.q_sample(px, pt, noise=physics_noise)
                peps = model(pxt, pt, py, plat)
                px0hat = torch.clamp(diffusion.x0_from_eps(pxt, pt, peps), -1, 1)
                pfake = denormalize_gauss(px0hat)
                if args.condition in ('hj', 'hj_pil'):
                    lhj = population_distribution_loss_v2(pfake, praw, plat)
                if args.condition in ('pil', 'hj_pil'):
                    lpil = pil_distribution_loss_v2(pfake, praw, pixel_mm=2.0)

            step += 1
            ramp = min(1.0, step / float(args.physics_warmup_steps)) if args.physics_warmup_steps > 0 else 1.0
            total = denoise + ramp * (args.lambda_hj*lhj + args.lambda_pil*lpil)

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, 0.999)

            rec = {
                'step': step,
                'epoch': epoch,
                'loss': float(total.item()),
                'denoise': float(denoise.item()),
                'hj': float(lhj.item()),
                'pil': float(lpil.item()),
                'physics_ramp': float(ramp),
                'physics_t_cut': phys_cut,
                'physics_batch_size': args.physics_batch_size if args.condition != 'base' else 0,
            }
            history.append(rec)
            if step % 20 == 0:
                print(json.dumps(rec), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if args.max_steps and step != args.max_steps:
        raise RuntimeError(f'Exact max-step budget not reached: requested={args.max_steps}, completed={step}')

    ck = {
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'condition': args.condition,
        'seed': args.seed,
        'steps': step,
        'diffusion_steps': args.diffusion_steps,
        'base_channels': args.base_channels,
        'lambda_hj': args.lambda_hj,
        'lambda_pil': args.lambda_pil,
        'physics_max_t_frac': args.physics_max_t_frac,
        'physics_batch_size': args.physics_batch_size,
        'generator_dropout': args.generator_dropout,
        'matched_mixed_diffusion_rng_across_conditions': True,
        'v2': True,
    }
    torch.save(ck, out/'generator.pt')
    (out/'training_history.json').write_text(json.dumps(history, indent=2)+'\n')
    (out/'run_config.json').write_text(json.dumps(vars(args) | {'device': str(device), 'steps_completed': step, 'effective_epochs': effective_epochs, 'steps_per_epoch': steps_per_epoch}, indent=2)+'\n')
    print(json.dumps({'condition': args.condition,'device': str(device),'steps': step,'checkpoint': str(out/'generator.pt')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
