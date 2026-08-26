from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from data import build_records, cache_records, MagnetogramDataset
from forecaster import FlareCNN, parameter_count
from metrics import all_metrics, region_bootstrap


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def temporal_even(df,n):
    if n<=0 or df.empty:return df.iloc[0:0].copy()
    if len(df)<=n:return df.copy()
    z=df.sort_values('t_rec').reset_index(drop=True);ids=np.unique(np.round(np.linspace(0,len(z)-1,n)).astype(int))
    if len(ids)<n:ids=np.r_[ids,[i for i in range(len(z)) if i not in set(ids)][:n-len(ids)]]
    return z.iloc[np.asarray(ids[:n],dtype=int)].copy()

def group_subset(df,per_group,pos_cap,seed):
    parts=[]
    for _,g in df.groupby('region_group_id',sort=True):
        pos=g[g.label_m1plus_24h.eq(1)];neg=g[g.label_m1plus_24h.eq(0)];kp=min(pos_cap,len(pos),per_group);kn=min(per_group-kp,len(neg))
        if kp==0:kn=min(per_group,len(neg))
        z=pd.concat([temporal_even(pos,kp),temporal_even(neg,kn)],ignore_index=True)
        if len(z)<per_group:
            rest=g[~g.sample_id.isin(z.sample_id)];z=pd.concat([z,temporal_even(rest,min(per_group-len(z),len(rest)))],ignore_index=True)
        parts.append(z)
    return pd.concat(parts,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)

def group_balanced_duplicates(real,n,seed):
    pos=real[real.label_m1plus_24h.eq(1)].copy();groups=sorted(pos.region_group_id.unique());rng=np.random.default_rng(seed);rows=[]
    if not groups or n<=0:return pos.iloc[0:0]
    for i in range(n):
        g=groups[i%len(groups)];z=pos[pos.region_group_id.eq(g)];rows.append(z.iloc[int(rng.integers(0,len(z)))].copy())
    out=pd.DataFrame(rows).reset_index(drop=True);out['sample_id']=[f'DUP_{seed}_{i:07d}' for i in range(len(out))];return out

class AugmentedDataset(Dataset):
    def __init__(self,real,synthetic_manifest=None,duplicate_count=0,seed=2026):
        self.real=MagnetogramDataset(real);self.items=[('real',i) for i in range(len(real))];self.dup=None;self.syn=None;self.syn_root=None
        if duplicate_count:
            d=group_balanced_duplicates(real,duplicate_count,seed);self.dup=MagnetogramDataset(d);self.items += [('dup',i) for i in range(len(d))]
        if synthetic_manifest:
            p=Path(synthetic_manifest);self.syn=pd.read_csv(p);self.syn_root=p.parent;self.items += [('syn',i) for i in range(len(self.syn))]
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        kind,j=self.items[i]
        if kind=='real':z=self.real[j];return z['x'],z['y']
        if kind=='dup':z=self.dup[j];return z['x'],z['y']
        r=self.syn.iloc[j];p=Path(str(r.array_path))
        if not p.exists():p=self.syn_root/'arrays'/p.name
        x=np.load(p).astype(np.float32);return torch.from_numpy(x)[None],torch.tensor(1.,dtype=torch.float32)

def collate_train(batch):return torch.stack([x for x,_ in batch]),torch.stack([y for _,y in batch])
def collate_eval(batch):return {'x':torch.stack([z['x'] for z in batch]),'y':torch.stack([z['y'] for z in batch]),'group':[z['group'] for z in batch],'sample_id':[z['sample_id'] for z in batch]}
@torch.no_grad()
def predict(model,loader,device):
    model.eval();rows=[]
    for b in loader:
        p=torch.sigmoid(model(b['x'].to(device))).cpu().numpy();y=b['y'].numpy();rows.extend({'sample_id':sid,'region_group_id':g,'y':int(yy),'p':float(pp)} for sid,g,yy,pp in zip(b['sample_id'],b['group'],y,p))
    return pd.DataFrame(rows)
def threshold(df):
    best=(-1e9,0.5,None)
    for t in np.linspace(.02,.98,97):
        m=all_metrics(df.y,df.p,float(t));s=m['tss'] if np.isfinite(m['tss']) else -1e9
        if s>best[0]:best=(s,float(t),m)
    return best[1],best[2]
class FocalBCE(nn.Module):
    def __init__(self,pos_weight,gamma=1.5):super().__init__();self.pos_weight=pos_weight;self.gamma=gamma
    def forward(self,logits,target):
        b=nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=self.pos_weight,reduction='none');p=torch.sigmoid(logits);pt=torch.where(target>.5,p,1-p);return (((1-pt).clamp_min(1e-6)**self.gamma)*b).mean()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--cache-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--arm',choices=['real','duplicate','synthetic'],required=True);ap.add_argument('--synthetic-manifest');ap.add_argument('--augmentation-count',type=int,default=250)
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--train-per-group',type=int,default=4);ap.add_argument('--val-per-group',type=int,default=6);ap.add_argument('--pos-cap',type=int,default=2)
    ap.add_argument('--width',type=int,default=48);ap.add_argument('--dropout',type=float,default=.2);ap.add_argument('--gamma',type=float,default=1.5);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--eval-every',type=int,default=300);ap.add_argument('--download-workers',type=int,default=16);ap.add_argument('--validation-bootstrap',type=int,default=1000);ap.add_argument('--evaluate-test',action='store_true')
    args=ap.parse_args();seed_all(args.seed);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    train=group_subset(build_records(args.evidence_dir,'train'),args.train_per_group,args.pos_cap,args.seed);val=group_subset(build_records(args.evidence_dir,'validation'),args.val_per_group,args.pos_cap,args.seed+1)
    train=cache_records(train,Path(args.cache_dir)/'train',args.download_workers);val=cache_records(val,Path(args.cache_dir)/'validation',args.download_workers);test=None
    if args.evaluate_test:test=cache_records(build_records(args.evidence_dir,'test'),Path(args.cache_dir)/'test',args.download_workers)
    if args.arm=='synthetic' and not args.synthetic_manifest:raise ValueError('--synthetic-manifest required')
    syn=args.synthetic_manifest if args.arm=='synthetic' else None;dup=args.augmentation_count if args.arm=='duplicate' else 0;tr=AugmentedDataset(train,syn,dup,args.seed)
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate_train,drop_last=True);vl=DataLoader(MagnetogramDataset(val),batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_eval)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=FlareCNN(width=args.width,dropout=args.dropout).to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    y=train.label_m1plus_24h.astype(int);base_pos=int(y.sum());base_neg=int((1-y).sum());fixed_pos_weight=torch.tensor(base_neg/max(1,base_pos),device=device,dtype=torch.float32);crit=FocalBCE(fixed_pos_weight,args.gamma)
    best=-1e9;best_state=None;history=[];step=0;iterator=iter(tl)
    while step<args.steps:
        try:x,yy=next(iterator)
        except StopIteration:iterator=iter(tl);x,yy=next(iterator)
        model.train();x=x.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True);loss=crit(model(x),yy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();step+=1
        if step%args.eval_every==0 or step==args.steps:
            vp=predict(model,vl,device);thr,vm=threshold(vp);score=vm['tss'] if np.isfinite(vm['tss']) else -1e9
            if score>best:best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            rec={'step':step,'loss':float(loss.item()),'threshold':thr,'validation':vm};history.append(rec);print(json.dumps(rec),flush=True)
    if best_state:model.load_state_dict(best_state)
    vp=predict(model,vl,device);thr,vm=threshold(vp);vp.to_csv(out/'validation_predictions.csv',index=False);vboot=region_bootstrap(vp,args.validation_bootstrap,args.seed,thr) if args.validation_bootstrap else None
    added=(len(pd.read_csv(syn)) if syn else dup);report={'arm':args.arm,'seed':args.seed,'locked_test':not args.evaluate_test,'architecture':{'width':args.width,'dropout':args.dropout,'loss':'focal','gamma':args.gamma,'lr':args.lr},'parameters':parameter_count(model),'fixed_steps':args.steps,'fixed_real_only_pos_weight':float(fixed_pos_weight.item()),'real_train_rows':len(train),'real_positive_rows':base_pos,'added_positive_rows':added,'total_train_items':len(tr),'validation_threshold':thr,'validation':vm,'validation_region_bootstrap':vboot,'history':history}
    if test is not None:
        tel=DataLoader(MagnetogramDataset(test),batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_eval);tp=predict(model,tel,device);tp.to_csv(out/'test_predictions.csv',index=False);report['test']=all_metrics(tp.y,tp.p,thr);report['test_region_bootstrap']=region_bootstrap(tp,2000,args.seed,thr)
    torch.save({'state_dict':model.state_dict(),'threshold':thr,'config':report['architecture'],'seed':args.seed},out/'model.pt');(out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');print(json.dumps(report,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
