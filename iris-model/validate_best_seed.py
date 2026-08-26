from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from benchmark_forecaster import seed_all, collate, group_balanced_subset, run_one
from data import build_records, cache_records, MagnetogramDataset


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True)
    ap.add_argument('--cache-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--train-per-group',type=int,default=8)
    ap.add_argument('--val-per-group',type=int,default=10)
    ap.add_argument('--pos-cap',type=int,default=4)
    ap.add_argument('--download-workers',type=int,default=16)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    train=group_balanced_subset(build_records(args.evidence_dir,'train'),args.train_per_group,args.pos_cap,args.seed)
    val=group_balanced_subset(build_records(args.evidence_dir,'validation'),args.val_per_group,args.pos_cap,args.seed+10000)
    print(json.dumps({'seed':args.seed,'locked_test':True,'train_rows':len(train),'train_groups':int(train.region_group_id.nunique()),'val_rows':len(val),'val_groups':int(val.region_group_id.nunique())},indent=2),flush=True)
    train=cache_records(train,Path(args.cache_dir)/'train',args.download_workers)
    val=cache_records(val,Path(args.cache_dir)/'validation',args.download_workers)
    tl=DataLoader(MagnetogramDataset(train),batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    vl=DataLoader(MagnetogramDataset(val),batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg={'name':f'w48_focal_seed{args.seed}','width':48,'dropout':0.20,'loss':'focal','gamma':1.5,'lr':3e-4}
    result=run_one(cfg,tl,vl,train,device,args.epochs,args.seed,out)
    (out/'seed_summary.json').write_text(json.dumps({'locked_test':True,'seed':args.seed,'result':result},indent=2,allow_nan=True)+'\n')
    print(json.dumps(result,indent=2,allow_nan=True),flush=True)

if __name__=='__main__':main()
