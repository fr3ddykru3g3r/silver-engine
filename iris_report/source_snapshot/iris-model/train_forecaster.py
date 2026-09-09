from __future__ import annotations

import argparse, json, math, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import build_records, deterministic_smoke_subset, cache_records, MagnetogramDataset
from forecaster import FlareCNN, parameter_count
from metrics import all_metrics, region_bootstrap


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def collate(batch):
    return {
        'x': torch.stack([b['x'] for b in batch]),
        'y': torch.stack([b['y'] for b in batch]),
        'group': [b['group'] for b in batch],
        'sample_id': [b['sample_id'] for b in batch],
    }


def predict(model, loader, device):
    model.eval(); rows=[]
    with torch.no_grad():
        for b in loader:
            p = torch.sigmoid(model(b['x'].to(device))).cpu().numpy()
            y = b['y'].numpy()
            for sid, g, yy, pp in zip(b['sample_id'], b['group'], y, p):
                rows.append({'sample_id':sid,'region_group_id':g,'y':int(yy),'p':float(pp)})
    return pd.DataFrame(rows)


def choose_tss_threshold(frame: pd.DataFrame):
    best=None
    for t in np.linspace(0.02,0.98,97):
        m=all_metrics(frame.y.values,frame.p.values,float(t))
        key=(m['tss'] if np.isfinite(m['tss']) else -999, -abs(t-0.5))
        if best is None or key>best[0]: best=(key,float(t),m)
    return best[1],best[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True)
    ap.add_argument('--cache-dir',default='cache/fits')
    ap.add_argument('--out-dir',default='outputs/forecaster')
    ap.add_argument('--epochs',type=int,default=15)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--download-workers',type=int,default=12)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache_dir)

    frames={p:build_records(args.evidence_dir,p) for p in ['train','validation','test']}
    if args.smoke:
        limits={'train':256,'validation':96,'test':96}
        frames={p:deterministic_smoke_subset(x,limits[p],args.seed+i) for i,(p,x) in enumerate(frames.items())}
        args.epochs=min(args.epochs,2)
    for p in frames:
        print(f'{p}: caching {len(frames[p])} FITS',flush=True)
        frames[p]=cache_records(frames[p],cache/p,args.download_workers)
        frames[p].to_csv(out/f'{p}_records.csv.gz',index=False,compression='gzip')

    ds={p:MagnetogramDataset(frames[p]) for p in frames}
    loaders={
        'train':DataLoader(ds['train'],batch_size=args.batch_size,shuffle=True,num_workers=args.workers,collate_fn=collate,pin_memory=torch.cuda.is_available()),
        'validation':DataLoader(ds['validation'],batch_size=args.batch_size,shuffle=False,num_workers=args.workers,collate_fn=collate),
        'test':DataLoader(ds['test'],batch_size=args.batch_size,shuffle=False,num_workers=args.workers,collate_fn=collate),
    }
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=FlareCNN().to(device)
    ytrain=frames['train'].label_m1plus_24h.astype(int)
    npos=int(ytrain.sum()); nneg=int((1-ytrain).sum())
    if npos==0: raise RuntimeError('No positive training examples')
    pos_weight=torch.tensor(nneg/npos,dtype=torch.float32,device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,args.epochs))

    best_auc=-math.inf; best_state=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.0; seen=0
        for b in loaders['train']:
            x=b['x'].to(device); y=b['y'].to(device)
            opt.zero_grad(set_to_none=True)
            loss=criterion(model(x),y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            total += float(loss.item())*len(y); seen += len(y)
        sched.step()
        val=predict(model,loaders['validation'],device)
        vm=all_metrics(val.y.values,val.p.values,0.5)
        auc=vm['auroc'] if np.isfinite(vm['auroc']) else -math.inf
        if auc>best_auc:
            best_auc=auc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rec={'epoch':epoch,'train_weighted_bce':total/max(1,seen),'val_auroc':vm['auroc'],'val_brier':vm['brier']}
        history.append(rec); print(json.dumps(rec),flush=True)
    if best_state is not None: model.load_state_dict(best_state)

    val=predict(model,loaders['validation'],device); test=predict(model,loaders['test'],device)
    threshold,val_metrics=choose_tss_threshold(val)
    test_metrics=all_metrics(test.y.values,test.p.values,threshold)
    boot=region_bootstrap(test,n_boot=100 if args.smoke else 2000,seed=args.seed,threshold=threshold)
    val.to_csv(out/'validation_predictions.csv',index=False); test.to_csv(out/'test_predictions.csv',index=False)
    torch.save({'state_dict':model.state_dict(),'seed':args.seed,'threshold':threshold},out/'flare_cnn.pt')
    report={
        'mode':'smoke' if args.smoke else 'full_real_only_baseline',
        'seed':args.seed,'device':str(device),'parameters':parameter_count(model),
        'train_rows':len(frames['train']),'train_positive_rows':npos,'pos_weight':float(pos_weight.item()),
        'threshold_selected_on_validation':threshold,'validation':val_metrics,'test':test_metrics,
        'test_region_cluster_bootstrap_95':boot,'history':history,
    }
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    print(json.dumps(report,indent=2,allow_nan=True),flush=True)

if __name__=='__main__': main()
