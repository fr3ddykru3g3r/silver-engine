from __future__ import annotations

import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

from cdr_models import CDRImageCNN,CDRImageBiLSTM,FeatureTransformer,count_parameters
from cdr_data import build_point_image_records,cache_image_sequences,CachedImageSequenceDataset,FeatureSequenceDataset
from data import cache_records,MagnetogramDataset
from metrics import all_metrics,region_bootstrap

PAPER_DL={
 'cnn':{'batch':16,'epochs':120,'optimizer':'sgd','lr':0.01},
 'cnn_bilstm':{'batch':16,'epochs':120,'optimizer':'sgd','lr':0.01},
 'transformer_2':{'batch':10,'epochs':150,'optimizer':'adam','lr':0.0001},
 'transformer_10':{'batch':10,'epochs':150,'optimizer':'adam','lr':0.0001},
}


def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)


def load_ds(a):
 parts=['train','validation']+(['test'] if a.evaluate_test else [])
 ds={};frames={}
 if a.model=='cnn':
  for j,p in enumerate(parts):
   per=a.train_per_group if p=='train' else a.eval_per_group
   r=build_point_image_records(a.evidence_dir,p,per,a.pos_cap,a.seed+j);r=cache_records(r,Path(a.cache_dir)/p,a.download_workers);ds[p]=MagnetogramDataset(r);frames[p]=r
 elif a.model=='cnn_bilstm':
  for j,p in enumerate(parts):
   per=a.seq_per_group if p=='train' else 1
   r,e=cache_image_sequences(a.evidence_dir,p,Path(a.cache_dir)/p,seq_len=a.seq_len,per_group=per,pos_cap=1,seed=a.seed+j,workers=a.download_workers);ds[p]=CachedImageSequenceDataset(r,e);frames[p]=e
 else:
  nf=2 if a.model.endswith('_2') else 10
  for j,p in enumerate(parts):
   per=a.train_per_group if p=='train' else a.eval_per_group
   z=FeatureSequenceDataset(a.evidence_dir,p,nf,per,a.pos_cap,a.seed+j);ds[p]=z;frames[p]=z.rows
 return ds,frames


def items_batch(ds,ids,dev):
 q=[ds[int(i)] for i in ids];return torch.stack([z['x'] for z in q]).to(dev),torch.stack([z['y'] for z in q]).to(dev),q

@torch.no_grad()
def predict(m,ds,dev,batch=64):
 m.eval();rows=[]
 for st in range(0,len(ds),batch):
  ids=range(st,min(st+batch,len(ds)));x,y,q=items_batch(ds,ids,dev);p=torch.sigmoid(m(x)).cpu().numpy()
  for z,pp in zip(q,p):rows.append({'sample_id':z['sample_id'],'region_group_id':z['group'],'y':int(z['y'].item()),'p':float(pp)})
 return pd.DataFrame(rows)

def choose(df):
 b=(-9,0.5,None)
 for t in np.linspace(.02,.98,97):
  m=all_metrics(df.y,df.p,float(t));s=m['tss'] if np.isfinite(m['tss']) else -9
  if s>b[0]:b=(s,float(t),m)
 return b[1],b[2]

def weights(labels):
 y=np.asarray(labels,dtype=int);N=len(y);n1=max(1,int(y.sum()));n0=max(1,N-n1)
 return N/(2*n0),N/(2*n1)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=list(PAPER_DL),required=True);ap.add_argument('--evidence-dir',required=True);ap.add_argument('--cache-dir',default='cache/cdr_dl');ap.add_argument('--out-dir',required=True)
 ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--train-per-group',type=int,default=8);ap.add_argument('--eval-per-group',type=int,default=10);ap.add_argument('--pos-cap',type=int,default=4);ap.add_argument('--seq-per-group',type=int,default=1);ap.add_argument('--seq-len',type=int,default=40);ap.add_argument('--download-workers',type=int,default=12)
 ap.add_argument('--epochs',type=int,default=0,help='0 = exact Table 4 epoch count');ap.add_argument('--max-train-items',type=int,default=0);ap.add_argument('--evaluate-test',action='store_true')
 a=ap.parse_args();seed_all(a.seed);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);cfg=PAPER_DL[a.model];epochs=a.epochs or cfg['epochs'];ds,frames=load_ds(a);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if a.model=='cnn':m=CDRImageCNN().to(dev)
 elif a.model=='cnn_bilstm':m=CDRImageBiLSTM().to(dev)
 else:m=FeatureTransformer(2 if a.model.endswith('_2') else 10,a.seq_len).to(dev)
 opt=torch.optim.SGD(m.parameters(),lr=cfg['lr'],momentum=.9) if cfg['optimizer']=='sgd' else torch.optim.Adam(m.parameters(),lr=cfg['lr'])
 labels=[int(ds['train'][i]['y'].item()) for i in range(len(ds['train']))];w0,w1=weights(labels);w0t=torch.tensor(w0,device=dev);w1t=torch.tensor(w1,device=dev);rng=np.random.default_rng(a.seed)
 best=-9;best_state=None;hist=[];batch=cfg['batch']
 for ep in range(1,epochs+1):
  order=rng.permutation(len(ds['train']));
  if a.max_train_items:order=order[:a.max_train_items]
  m.train();losses=[]
  for st in range(0,len(order),batch):
   ids=order[st:st+batch]
   if len(ids)<2:continue
   x,y,_=items_batch(ds['train'],ids,dev);logit=m(x);raw=nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none');ww=torch.where(y>0.5,w1t,w0t);loss=(raw*ww).mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5.0);opt.step();losses.append(float(loss.item()))
  val=predict(m,ds['validation'],dev);vm=all_metrics(val.y,val.p,.5);score=vm['tss'] if np.isfinite(vm['tss']) else -9
  if score>best:best=score;best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  rec={'epoch':ep,'loss':float(np.mean(losses)) if losses else None,'validation_at_0.5':vm};hist.append(rec)
  if ep==1 or ep%10==0:print(json.dumps(rec),flush=True)
 if best_state:m.load_state_dict(best_state)
 val=predict(m,ds['validation'],dev);thr,vmopt=choose(val);vm05=all_metrics(val.y,val.p,.5);val.to_csv(out/'validation_predictions.csv',index=False)
 rep={'model':a.model,'paper_training':cfg,'epochs_run':epochs,'class_weights':{'negative':w0,'positive':w1},'seed':a.seed,'parameters':count_parameters(m),'train_items':len(ds['train']),'validation_items':len(ds['validation']),'validation_at_0.5':vm05,'validation_selected_threshold':thr,'validation_selected':vmopt,'history':hist,'test_locked':not a.evaluate_test}
 if a.evaluate_test:
  te=predict(m,ds['test'],dev);te.to_csv(out/'test_predictions.csv',index=False);rep['test']=all_metrics(te.y,te.p,thr);rep['test_region_bootstrap']=region_bootstrap(te,2000,a.seed,thr)
 torch.save({'state_dict':m.state_dict(),'model':a.model,'seed':a.seed,'threshold':thr,'seq_len':a.seq_len},out/'model.pt');(out/'metrics.json').write_text(json.dumps(rep,indent=2,allow_nan=True)+'\n');print(json.dumps(rep,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
