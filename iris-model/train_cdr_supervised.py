from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

from cdr_models import CDRImageCNN, CDRImageBiLSTM, FeatureTransformer, count_parameters
from cdr_data import build_point_image_records, cache_native_image_sequences, FeatureSequenceDataset
from data import cache_records, MagnetogramDataset
from metrics import all_metrics, region_bootstrap


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_dataset(args, partition:str, offset:int=0):
    if args.model=='cnn':
        per=args.train_per_group if partition=='train' else args.eval_per_group
        rec=build_point_image_records(args.evidence_dir,partition,per,args.pos_cap,args.seed+offset)
        rec=cache_records(rec,Path(args.cache_dir)/partition,args.download_workers)
        return MagnetogramDataset(rec),rec
    if args.model=='cnn_bilstm':
        per=args.seq_per_group if partition=='train' else 1
        maxg=args.seq_max_groups_train if partition=='train' else args.seq_max_groups_eval
        ds,end,_=cache_native_image_sequences(args.evidence_dir,partition,Path(args.cache_dir)/partition,
                                               per_group=per,pos_cap=1,seed=args.seed+offset,
                                               workers=args.download_workers,max_groups=maxg)
        return ds,end
    nf=2 if args.model=='transformer_2' else 10
    per=args.train_per_group if partition=='train' else args.eval_per_group
    ds=FeatureSequenceDataset(args.evidence_dir,partition,n_features=nf,per_group=per,pos_cap=args.pos_cap,seed=args.seed+offset)
    return ds,ds.rows


def stack(dataset, indices, device):
    z=[dataset[int(i)] for i in indices]
    return torch.stack([a['x'] for a in z]).to(device),torch.stack([a['y'] for a in z]).to(device)


@torch.no_grad()
def predict(model,dataset,device,batch=64):
    rows=[];model.eval()
    for st in range(0,len(dataset),batch):
        ids=range(st,min(st+batch,len(dataset)));z=[dataset[i] for i in ids]
        p=torch.sigmoid(model(torch.stack([a['x'] for a in z]).to(device))).cpu().numpy()
        rows.extend({'sample_id':a['sample_id'],'region_group_id':a['group'],'y':int(a['y'].item()),'p':float(pp)} for a,pp in zip(z,p))
    import pandas as pd
    return pd.DataFrame(rows)


def threshold(frame):
    best=(-1e9,0.5,None)
    for t in np.linspace(.02,.98,97):
        m=all_metrics(frame.y.values,frame.p.values,float(t));s=m['tss'] if np.isfinite(m['tss']) else -1e9
        if s>best[0]:best=(s,float(t),m)
    return best[1],best[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['cnn','cnn_bilstm','transformer_2','transformer_10'],required=True)
    ap.add_argument('--evidence-dir',required=True);ap.add_argument('--cache-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--train-per-group',type=int,default=8);ap.add_argument('--eval-per-group',type=int,default=10);ap.add_argument('--pos-cap',type=int,default=4);ap.add_argument('--seq-per-group',type=int,default=1)
    ap.add_argument('--seq-max-groups-train',type=int,default=0);ap.add_argument('--seq-max-groups-eval',type=int,default=0)
    ap.add_argument('--download-workers',type=int,default=12);ap.add_argument('--evaluate-test',action='store_true')
    args=ap.parse_args();seed_all(args.seed);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    train,_=make_dataset(args,'train',0);val,_=make_dataset(args,'validation',1)
    test=None
    if args.evaluate_test:test,_=make_dataset(args,'test',2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model=='cnn': model=CDRImageCNN().to(device);paper_epochs=120;batch=16;lr=.01;opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
    elif args.model=='cnn_bilstm': model=CDRImageBiLSTM().to(device);paper_epochs=120;batch=16;lr=.01;opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
    elif args.model=='transformer_2': model=FeatureTransformer(2,seq_len=40).to(device);paper_epochs=150;batch=10;lr=1e-4;opt=torch.optim.Adam(model.parameters(),lr=lr)
    else: model=FeatureTransformer(10,seq_len=40).to(device);paper_epochs=150;batch=10;lr=1e-4;opt=torch.optim.Adam(model.parameters(),lr=lr)
    epochs=args.epochs or paper_epochs
    ys=np.array([int(train[i]['y'].item()) for i in range(len(train))]);npos=max(1,int(ys.sum()));nneg=max(1,len(ys)-npos)
    pos_weight=torch.tensor(nneg/npos,dtype=torch.float32,device=device);crit=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng=np.random.default_rng(args.seed);history=[];best=-1e9;state=None
    for e in range(1,epochs+1):
        order=rng.permutation(len(train));model.train();tot=0.;seen=0
        for st in range(0,len(order),batch):
            ids=order[st:st+batch];x,y=stack(train,ids,device);opt.zero_grad(set_to_none=True);loss=crit(model(x),y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();tot+=float(loss.item())*len(ids);seen+=len(ids)
        vp=predict(model,val,device);thr,vm=threshold(vp);score=vm['tss'] if np.isfinite(vm['tss']) else -1e9
        if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        r={'epoch':e,'loss':tot/max(1,seen),'validation_threshold':thr,'validation':vm};history.append(r)
        print(json.dumps(r),flush=True)
    if state:model.load_state_dict(state)
    vp=predict(model,val,device);thr,vm=threshold(vp);vm05=all_metrics(vp.y,vp.p,.5);vp.to_csv(out/'validation_predictions.csv',index=False)
    report={'model':args.model,'seed':args.seed,'device':str(device),'parameters':count_parameters(model),'paper_default_epochs':paper_epochs,'epochs_run':epochs,'paper_optimizer':'SGD' if args.model.startswith('cnn') else 'Adam','paper_lr':lr,'paper_batch':batch,'seq_max_groups_train':args.seq_max_groups_train,'seq_max_groups_eval':args.seq_max_groups_eval,'train_items':len(train),'validation_items':len(val),'validation_at_0.5':vm05,'validation_selected_threshold':thr,'validation_selected':vm,'history':history,'test_locked':not args.evaluate_test}
    if test is not None:
        tp=predict(model,test,device);tp.to_csv(out/'test_predictions.csv',index=False);report['test']=all_metrics(tp.y,tp.p,thr);report['test_region_bootstrap']=region_bootstrap(tp,2000,args.seed,thr);report['test_items']=len(test)
    torch.save({'state_dict':model.state_dict(),'model':args.model,'threshold':thr,'seed':args.seed},out/'model.pt')
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');print(json.dumps(report,indent=2,allow_nan=True),flush=True)

if __name__=='__main__':main()
