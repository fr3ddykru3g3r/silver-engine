from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from data import build_records, cache_records, MagnetogramDataset, deterministic_smoke_subset
from forecaster import FlareCNN, parameter_count
from metrics import all_metrics, region_bootstrap


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def group_balanced_duplicates(real: pd.DataFrame,n:int,seed:int):
    pos=real[real.label_m1plus_24h.eq(1)].copy(); groups=sorted(pos.region_group_id.unique())
    if not groups or n<=0:return pos.iloc[0:0]
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        g=groups[i%len(groups)]; z=pos[pos.region_group_id.eq(g)]
        rows.append(z.iloc[int(rng.integers(0,len(z)))].copy())
    out=pd.DataFrame(rows).reset_index(drop=True); out['sample_id']=[f'DUP_{seed}_{i:07d}' for i in range(len(out))]
    return out


class DownstreamTrainDataset(Dataset):
    def __init__(self,real_records:pd.DataFrame,synthetic_manifest:str|None=None,duplicate_count:int=0,seed:int=2026):
        self.real=MagnetogramDataset(real_records); self.items=[('real',i) for i in range(len(real_records))]
        self.synthetic=None
        if duplicate_count:
            dup=group_balanced_duplicates(real_records,duplicate_count,seed)
            self.dup=MagnetogramDataset(dup); self.items += [('dup',i) for i in range(len(dup))]
        else:self.dup=None
        if synthetic_manifest:
            self.synthetic=pd.read_csv(synthetic_manifest)
            self.items += [('syn',i) for i in range(len(self.synthetic))]
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        kind,j=self.items[i]
        if kind=='real':
            b=self.real[j]; return {'x':b['x'],'y':b['y']}
        if kind=='dup':
            b=self.dup[j]; return {'x':b['x'],'y':b['y']}
        r=self.synthetic.iloc[j]; x=np.load(r.array_path).astype(np.float32)
        return {'x':torch.from_numpy(x)[None],'y':torch.tensor(1.0,dtype=torch.float32)}


def collate_train(b):return {'x':torch.stack([z['x'] for z in b]),'y':torch.stack([z['y'] for z in b])}
def collate_eval(b):return {'x':torch.stack([z['x'] for z in b]),'y':torch.stack([z['y'] for z in b]),'group':[z['group'] for z in b],'sample_id':[z['sample_id'] for z in b]}


def predict(model,loader,device):
    model.eval(); rows=[]
    with torch.no_grad():
        for b in loader:
            p=torch.sigmoid(model(b['x'].to(device))).cpu().numpy(); y=b['y'].numpy()
            for sid,g,yy,pp in zip(b['sample_id'],b['group'],y,p):rows.append({'sample_id':sid,'region_group_id':g,'y':int(yy),'p':float(pp)})
    return pd.DataFrame(rows)


def choose_tss_threshold(df):
    best=(-999,0.5,None)
    for t in np.linspace(.02,.98,97):
        m=all_metrics(df.y.values,df.p.values,float(t)); score=m['tss'] if np.isfinite(m['tss']) else -999
        if score>best[0]:best=(score,float(t),m)
    return best[1],best[2]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',default='cache/downstream'); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--arm',choices=['real','duplicate','synthetic'],required=True); ap.add_argument('--synthetic-manifest'); ap.add_argument('--augmentation-count',type=int,default=0)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--max-steps',type=int,default=0); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); seed_all(args.seed); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    frames={p:build_records(args.evidence_dir,p) for p in ['train','validation','test']}
    if args.smoke:
        lim={'train':192,'validation':64,'test':64}; frames={p:deterministic_smoke_subset(x,lim[p],args.seed+i) for i,(p,x) in enumerate(frames.items())}; args.epochs=1; args.max_steps=4
    for p in frames:frames[p]=cache_records(frames[p],Path(args.cache_dir)/p,workers=10)
    if args.arm=='synthetic' and not args.synthetic_manifest:raise ValueError('--synthetic-manifest required')
    if args.arm=='duplicate' and args.augmentation_count<=0:raise ValueError('--augmentation-count >0 required')
    syn=args.synthetic_manifest if args.arm=='synthetic' else None; dup=args.augmentation_count if args.arm=='duplicate' else 0
    tr=DownstreamTrainDataset(frames['train'],syn,dup,args.seed)
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate_train)
    ev={p:DataLoader(MagnetogramDataset(frames[p]),batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=collate_eval) for p in ['validation','test']}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=FlareCNN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    y_real=frames['train'].label_m1plus_24h.astype(int); base_pos=int(y_real.sum()); base_neg=int((1-y_real).sum()); added=(len(pd.read_csv(syn)) if syn else dup)
    # Weight against the ACTUAL augmented positive count so all arms use the same loss definition.
    pos_count=base_pos+added; pos_weight=torch.tensor(base_neg/max(1,pos_count),device=device,dtype=torch.float32); crit=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best=-1; best_state=None; history=[]; step=0
    for e in range(1,args.epochs+1):
        model.train(); total=0.; seen=0
        for b in tl:
            x=b['x'].to(device); y=b['y'].to(device); opt.zero_grad(set_to_none=True); loss=crit(model(x),y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
            total+=float(loss)*len(y);seen+=len(y);step+=1
            if args.max_steps and step>=args.max_steps:break
        val=predict(model,ev['validation'],device); vm=all_metrics(val.y,val.p,.5); auc=vm['auroc'] if np.isfinite(vm['auroc']) else -1
        if auc>best:best=auc;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'epoch':e,'step':step,'loss':total/max(1,seen),'val_auroc':vm['auroc']});print(json.dumps(history[-1]),flush=True)
        if args.max_steps and step>=args.max_steps:break
    if best_state:model.load_state_dict(best_state)
    val=predict(model,ev['validation'],device);test=predict(model,ev['test'],device);thr,vm=choose_tss_threshold(val);tm=all_metrics(test.y,test.p,thr);boot=region_bootstrap(test,100 if args.smoke else 2000,args.seed,thr)
    val.to_csv(out/'validation_predictions.csv',index=False);test.to_csv(out/'test_predictions.csv',index=False);torch.save(model.state_dict(),out/'forecaster.pt')
    rep={'arm':args.arm,'seed':args.seed,'parameters':parameter_count(model),'real_train_rows':len(frames['train']),'base_positive_rows':base_pos,'added_positive_rows':added,'total_train_items':len(tr),'threshold_validation':thr,'validation':vm,'test':tm,'test_region_bootstrap':boot,'history':history}
    (out/'metrics.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2),flush=True)
if __name__=='__main__':main()
