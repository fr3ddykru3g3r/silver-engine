from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from cdr import (PAPER_CDR_PRESETS, IndexReplayBuffer, Experience, choose_actions,
                 immediate_rewards, cdr_loss, epsilon_after_episode, reward_counts)
from cdr_models import CDRImageCNN, CDRImageBiLSTM, FeatureTransformer, count_parameters
from cdr_data import (build_point_image_records, cache_image_sequences,
                      CachedImageSequenceDataset, FeatureSequenceDataset, temporal_group_cap)
from data import cache_records, MagnetogramDataset
from metrics import all_metrics, region_bootstrap


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def stack_items(dataset, indices, device):
    items=[dataset[int(i)] for i in indices]
    x=torch.stack([z['x'] for z in items]).to(device)
    y=torch.stack([z['y'] for z in items]).to(device)
    return x,y


@torch.no_grad()
def predict(model,dataset,device,batch_size=64):
    model.eval();rows=[]
    for st in range(0,len(dataset),batch_size):
        ids=list(range(st,min(st+batch_size,len(dataset))))
        items=[dataset[i] for i in ids]
        x=torch.stack([z['x'] for z in items]).to(device)
        p=torch.sigmoid(model(x)).cpu().numpy()
        for z,pp in zip(items,p):
            rows.append({'sample_id':z['sample_id'],'region_group_id':z['group'],'y':int(z['y'].item()),'p':float(pp)})
    return pd.DataFrame(rows)


def choose_val_threshold(frame):
    best=(-1e9,0.5,None)
    for t in np.linspace(.02,.98,97):
        m=all_metrics(frame.y.values,frame.p.values,float(t))
        score=m['tss'] if np.isfinite(m['tss']) else -1e9
        if score>best[0]:best=(score,float(t),m)
    return best[1],best[2]


def load_datasets(args):
    if args.model=='cdr_cnn':
        frames={}
        ds={}
        for j,p in enumerate(['train','validation'] + (['test'] if args.evaluate_test else [])):
            per=args.train_per_group if p=='train' else args.eval_per_group
            rec=build_point_image_records(args.evidence_dir,p,per,args.pos_cap,args.seed+j)
            rec=cache_records(rec,Path(args.cache_dir)/p,args.download_workers)
            frames[p]=rec;ds[p]=MagnetogramDataset(rec)
        return ds,frames
    if args.model=='cdr_cnn_bilstm':
        ds={};frames={}
        for j,p in enumerate(['train','validation'] + (['test'] if args.evaluate_test else [])):
            per=args.seq_per_group if p=='train' else 1
            rec,end=cache_image_sequences(args.evidence_dir,p,Path(args.cache_dir)/p,
                                          seq_len=args.seq_len,per_group=per,pos_cap=1,
                                          seed=args.seed+j,workers=args.download_workers)
            ds[p]=CachedImageSequenceDataset(rec,end);frames[p]=end
        return ds,frames
    # Feature sequence models require the CDR-enriched evidence artifact.
    nf=2 if args.model=='cdr_transformer_2' else 10
    ds={};frames={}
    for j,p in enumerate(['train','validation'] + (['test'] if args.evaluate_test else [])):
        per=args.train_per_group if p=='train' else args.eval_per_group
        z=FeatureSequenceDataset(args.evidence_dir,p,n_features=nf,per_group=per,pos_cap=args.pos_cap,seed=args.seed+j)
        ds[p]=z;frames[p]=z.rows
    return ds,frames


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['cdr_cnn','cdr_cnn_bilstm','cdr_transformer_2','cdr_transformer_10'],required=True)
    ap.add_argument('--evidence-dir',required=True);ap.add_argument('--cache-dir',default='cache/cdr');ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--train-per-group',type=int,default=8);ap.add_argument('--eval-per-group',type=int,default=10);ap.add_argument('--pos-cap',type=int,default=4)
    ap.add_argument('--seq-per-group',type=int,default=1);ap.add_argument('--seq-len',type=int,default=24)
    ap.add_argument('--download-workers',type=int,default=12);ap.add_argument('--grad-clip',type=float,default=5.0)
    ap.add_argument('--paper-update-rate',action='store_true',help='Perform one replay update per encountered state, matching the paper description. Default uses one update per encounter mini-batch for CPU feasibility.')
    ap.add_argument('--evaluate-test',action='store_true',help='Touch the locked test partition only after model/reward choices are frozen.')
    ap.add_argument('--episodes',type=int,default=0,help='Override paper episode count; 0 uses Table 5.')
    ap.add_argument('--max-encounters',type=int,default=0,help='Engineering smoke limit per episode; 0 means full selected training set.')
    args=ap.parse_args();seed_all(args.seed)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    preset='cdr_transformer' if args.model.startswith('cdr_transformer') else args.model
    cfg=PAPER_CDR_PRESETS[preset];episodes=args.episodes or cfg.episodes
    ds,frames=load_datasets(args)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model=='cdr_cnn':model=CDRImageCNN().to(device)
    elif args.model=='cdr_cnn_bilstm':model=CDRImageBiLSTM().to(device)
    else:model=FeatureTransformer(2 if args.model.endswith('_2') else 10,seq_len=args.seq_len).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    replay=IndexReplayBuffer(cfg.replay_size,args.seed)
    rng=np.random.default_rng(args.seed)
    history=[];best_score=-1e9;best_state=None
    train_indices=np.arange(len(ds['train']))
    for ep in range(episodes):
        epsilon=epsilon_after_episode(cfg,ep)
        order=rng.permutation(train_indices)
        if args.max_encounters:order=order[:args.max_encounters]
        ep_loss=[];ep_conf={'tp':0,'tn':0,'fp':0,'fn':0};updates=0
        encounter_chunk=max(1,cfg.batch_size)
        for st in range(0,len(order),encounter_chunk):
            ids=order[st:st+encounter_chunk]
            model.eval()
            with torch.no_grad():
                x,y=stack_items(ds['train'],ids,device);logits=model(x)
                actions=choose_actions(logits,epsilon,rng);rewards=immediate_rewards(y,actions,cfg)
            rc=reward_counts(y,actions)
            for k in ep_conf:ep_conf[k]+=rc[k]
            for idx,a,r,yy in zip(ids,actions.cpu(),rewards.cpu(),y.cpu()):
                replay.append(Experience(int(idx),int(a.item()),float(r.item()),int(yy.item())))
            n_updates=len(ids) if args.paper_update_rate else 1
            for _ in range(n_updates):
                if len(replay)<cfg.batch_size:break
                batch=replay.sample(cfg.batch_size);bids=[e.index for e in batch]
                bx,by=stack_items(ds['train'],bids,device)
                ba=torch.tensor([e.action for e in batch],device=device,dtype=torch.long)
                br=torch.tensor([e.reward for e in batch],device=device,dtype=torch.float32)
                model.train();opt.zero_grad(set_to_none=True);bl=model(bx);loss=cdr_loss(bl,ba,br)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);opt.step()
                ep_loss.append(float(loss.item()));updates+=1
        val=predict(model,ds['validation'],device)
        # Paper reports threshold 0.5; we preserve it and separately report the
        # validation-selected threshold used by our primary forecasting protocol.
        vm05=all_metrics(val.y.values,val.p.values,0.5);thr,vmopt=choose_val_threshold(val)
        score=vm05['tss'] if np.isfinite(vm05['tss']) else -1e9
        if score>best_score:
            best_score=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rec={'episode':ep+1,'epsilon':epsilon,'replay_size':len(replay),'updates':updates,
             'mean_cdr_loss':float(np.mean(ep_loss)) if ep_loss else None,'encounter_confusion':ep_conf,
             'validation_at_0.5':vm05,'validation_opt_threshold':thr,'validation_opt':vmopt}
        history.append(rec);print(json.dumps(rec),flush=True)
    if best_state is not None:model.load_state_dict(best_state)
    val=predict(model,ds['validation'],device);thr,vm=choose_val_threshold(val);val05=all_metrics(val.y,val.p,0.5)
    val.to_csv(out/'validation_predictions.csv',index=False)
    report={'model':args.model,'seed':args.seed,'device':str(device),'parameters':count_parameters(model),
            'paper_config':cfg.__dict__,'episodes_run':episodes,'paper_update_rate':bool(args.paper_update_rate),
            'train_items':len(ds['train']),'validation_items':len(ds['validation']),
            'validation_at_0.5':val05,'validation_selected_threshold':thr,'validation_selected':vm,
            'history':history,'test_locked':not args.evaluate_test}
    if args.evaluate_test:
        test=predict(model,ds['test'],device);test.to_csv(out/'test_predictions.csv',index=False)
        tm=all_metrics(test.y.values,test.p.values,thr);boot=region_bootstrap(test,2000,args.seed,thr)
        report['test']=tm;report['test_region_bootstrap']=boot;report['test_items']=len(ds['test'])
    torch.save({'state_dict':model.state_dict(),'model':args.model,'seed':args.seed,'config':cfg.__dict__,'threshold':thr},out/'cdr_model.pt')
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    print(json.dumps(report,indent=2,allow_nan=True),flush=True)

if __name__=='__main__':main()
