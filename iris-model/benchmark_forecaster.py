from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import build_records, cache_records, MagnetogramDataset
from forecaster import FlareCNN, parameter_count
from metrics import all_metrics


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collate(batch):
    return {'x':torch.stack([b['x'] for b in batch]),
            'y':torch.stack([b['y'] for b in batch]),
            'group':[b['group'] for b in batch],
            'sample_id':[b['sample_id'] for b in batch]}


def temporal_even(df:pd.DataFrame,n:int):
    if n<=0 or df.empty: return df.iloc[0:0].copy()
    if len(df)<=n: return df.copy()
    z=df.sort_values('t_rec').reset_index(drop=True)
    idx=np.unique(np.round(np.linspace(0,len(z)-1,n)).astype(int))
    if len(idx)<n:
        remaining=[i for i in range(len(z)) if i not in set(idx)]
        idx=np.concatenate([idx,np.asarray(remaining[:n-len(idx)])])
    return z.iloc[np.sort(idx[:n])].copy()


def group_balanced_subset(df:pd.DataFrame,per_group:int,pos_cap:int,seed:int):
    parts=[]
    for gi,(gid,g) in enumerate(df.groupby('region_group_id',sort=True)):
        pos=g[g.label_m1plus_24h.eq(1)]
        neg=g[g.label_m1plus_24h.eq(0)]
        kp=min(pos_cap,len(pos),per_group)
        kn=min(per_group-kp,len(neg))
        # If a nonflaring region has no positives, use the full per-group budget on negatives.
        if kp==0: kn=min(per_group,len(neg))
        p=temporal_even(pos,kp)
        q=temporal_even(neg,kn)
        z=pd.concat([p,q],ignore_index=True)
        if len(z)<per_group:
            rest=g[~g.sample_id.isin(z.sample_id)]
            z=pd.concat([z,temporal_even(rest,min(per_group-len(z),len(rest)))],ignore_index=True)
        parts.append(z)
    out=pd.concat(parts,ignore_index=True)
    return out.sample(frac=1,random_state=seed).reset_index(drop=True)


def predict(model,loader,device):
    model.eval(); rows=[]
    with torch.no_grad():
        for b in loader:
            p=torch.sigmoid(model(b['x'].to(device))).cpu().numpy()
            y=b['y'].numpy()
            rows.extend({'sample_id':sid,'region_group_id':g,'y':int(yy),'p':float(pp)}
                        for sid,g,yy,pp in zip(b['sample_id'],b['group'],y,p))
    return pd.DataFrame(rows)


def choose_tss_threshold(frame):
    best=None
    for t in np.linspace(0.02,0.98,97):
        m=all_metrics(frame.y.values,frame.p.values,float(t))
        key=(m['tss'] if np.isfinite(m['tss']) else -999,
             m['hss'] if np.isfinite(m['hss']) else -999,
             -abs(float(t)-0.5))
        if best is None or key>best[0]: best=(key,float(t),m)
    return best[1],best[2]


class FocalBCE(nn.Module):
    def __init__(self,pos_weight:torch.Tensor,gamma:float=1.5):
        super().__init__(); self.pos_weight=pos_weight; self.gamma=gamma
    def forward(self,logits,target):
        bce=nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=self.pos_weight,reduction='none')
        p=torch.sigmoid(logits)
        pt=torch.where(target>0.5,p,1-p)
        return (((1-pt).clamp_min(1e-6)**self.gamma)*bce).mean()


def run_one(cfg,train_loader,val_loader,train_frame,device,epochs,seed,out):
    seed_all(seed)
    model=FlareCNN(width=cfg['width'],dropout=cfg['dropout']).to(device)
    y=train_frame.label_m1plus_24h.astype(int)
    npos=int(y.sum()); nneg=int((1-y).sum())
    pw=torch.tensor(nneg/max(1,npos),dtype=torch.float32,device=device)
    criterion=FocalBCE(pw,cfg.get('gamma',1.5)) if cfg['loss']=='focal' else nn.BCEWithLogitsLoss(pos_weight=pw)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg.get('weight_decay',1e-4))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs))
    best_key=None; best_state=None; history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=0.; seen=0
        for b in train_loader:
            x=b['x'].to(device); yy=b['y'].to(device)
            opt.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,yy)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            total+=float(loss.item())*len(yy); seen+=len(yy)
        sched.step()
        val=predict(model,val_loader,device); thr,vm=choose_tss_threshold(val)
        key=(vm['tss'] if np.isfinite(vm['tss']) else -999,
             vm['auroc'] if np.isfinite(vm['auroc']) else -999,
             -(vm['brier'] if np.isfinite(vm['brier']) else 999))
        if best_key is None or key>best_key:
            best_key=key; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rec={'epoch':epoch,'train_loss':total/max(1,seen),'threshold':thr,**{f'val_{k}':v for k,v in vm.items()}}
        history.append(rec); print(json.dumps({'config':cfg['name'],**rec}),flush=True)
    model.load_state_dict(best_state)
    val=predict(model,val_loader,device); thr,vm=choose_tss_threshold(val)
    d=out/cfg['name']; d.mkdir(parents=True,exist_ok=True)
    val.to_csv(d/'validation_predictions.csv',index=False)
    torch.save({'state_dict':model.state_dict(),'threshold':thr,'config':cfg,'seed':seed},d/'model.pt')
    (d/'history.json').write_text(json.dumps(history,indent=2,allow_nan=True)+'\n')
    return {'name':cfg['name'],'seed':seed,'parameters':parameter_count(model),'threshold':thr,**vm,'config':cfg}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True)
    ap.add_argument('--cache-dir',default='cache/benchmark')
    ap.add_argument('--out-dir',default='outputs/benchmark')
    ap.add_argument('--train-per-group',type=int,default=10)
    ap.add_argument('--val-per-group',type=int,default=12)
    ap.add_argument('--pos-cap',type=int,default=5)
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--download-workers',type=int,default=16)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    # Deliberately do not read the test partition during model selection.
    train=group_balanced_subset(build_records(args.evidence_dir,'train'),args.train_per_group,args.pos_cap,args.seed)
    val=group_balanced_subset(build_records(args.evidence_dir,'validation'),args.val_per_group,args.pos_cap,args.seed+1)
    print(json.dumps({'locked_test':True,'train_rows':len(train),'train_groups':train.region_group_id.nunique(),
                      'train_positive_rows':int(train.label_m1plus_24h.sum()),'val_rows':len(val),
                      'val_groups':val.region_group_id.nunique(),'val_positive_rows':int(val.label_m1plus_24h.sum())},indent=2),flush=True)
    train=cache_records(train,Path(args.cache_dir)/'train',args.download_workers)
    val=cache_records(val,Path(args.cache_dir)/'validation',args.download_workers)
    train.to_csv(out/'train_records.csv.gz',index=False,compression='gzip'); val.to_csv(out/'validation_records.csv.gz',index=False,compression='gzip')
    ds_train=MagnetogramDataset(train); ds_val=MagnetogramDataset(val)
    tl=DataLoader(ds_train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    vl=DataLoader(ds_val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,collate_fn=collate)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs=[
        {'name':'w32_bce','width':32,'dropout':0.20,'loss':'bce','lr':3e-4},
        {'name':'w48_bce','width':48,'dropout':0.20,'loss':'bce','lr':3e-4},
        {'name':'w48_lowdrop','width':48,'dropout':0.10,'loss':'bce','lr':3e-4},
        {'name':'w48_focal','width':48,'dropout':0.20,'loss':'focal','gamma':1.5,'lr':3e-4},
    ]
    results=[run_one(c,tl,vl,train,device,args.epochs,args.seed,out) for c in configs]
    lb=pd.DataFrame([{k:v for k,v in r.items() if k!='config'} for r in results]).sort_values(['tss','auroc','brier'],ascending=[False,False,True])
    lb.to_csv(out/'leaderboard.csv',index=False)
    best=results[int(np.argmax([((r['tss'] if np.isfinite(r['tss']) else -999)*1000 + (r['auroc'] if np.isfinite(r['auroc']) else -999)) for r in results]))]
    summary={'device':str(device),'locked_test':True,'selection_metric':'validation TSS, then AUROC/Brier','best':best,'all':results}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=True),flush=True)

if __name__=='__main__': main()
