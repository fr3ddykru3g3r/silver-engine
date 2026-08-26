from __future__ import annotations

import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import shap

from cdr_models import FeatureTransformer
from cdr_data import FeatureSequenceDataset,FEATURES2,FEATURES10


def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)


def tensor_subset(ds,ids):
    items=[ds[int(i)] for i in ids]
    return torch.stack([z['x'] for z in items]),items


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--evidence-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--model',choices=['transformer_2','transformer_10'],required=True);ap.add_argument('--partition',choices=['validation','test'],default='validation');ap.add_argument('--max-samples',type=int,default=128);ap.add_argument('--background',type=int,default=64);ap.add_argument('--seed',type=int,default=2026)
    a=ap.parse_args();seed_all(a.seed);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);nf=2 if a.model.endswith('_2') else 10;names=FEATURES2 if nf==2 else FEATURES10
    train=FeatureSequenceDataset(a.evidence_dir,'train',nf,per_group=2,pos_cap=1,seed=a.seed)
    target=FeatureSequenceDataset(a.evidence_dir,a.partition,nf,per_group=1,pos_cap=1,seed=a.seed+1)
    rng=np.random.default_rng(a.seed);bid=rng.choice(len(train),size=min(a.background,len(train)),replace=False);tid=rng.choice(len(target),size=min(a.max_samples,len(target)),replace=False)
    bg,_=tensor_subset(train,bid);x,items=tensor_subset(target,tid)
    ck=torch.load(a.checkpoint,map_location='cpu');seq_len=int(ck.get('seq_len',x.shape[1]));m=FeatureTransformer(nf,seq_len=seq_len);m.load_state_dict(ck['state_dict']);m.eval()
    # GradientExplainer yields SHAP attributions over time x feature. We aggregate
    # absolute values across time and samples for global importance, matching the
    # paper's interpretation of a feature across all temporal positions.
    explainer=shap.GradientExplainer(m,bg)
    sv=explainer.shap_values(x)
    if isinstance(sv,list):sv=sv[0]
    sv=np.asarray(sv)
    if sv.ndim==4 and sv.shape[-1]==1:sv=sv[...,0]
    if sv.shape[:2]!=(len(x),x.shape[1]):
        raise RuntimeError(f'unexpected SHAP shape {sv.shape} for input {tuple(x.shape)}')
    global_mean_abs=np.mean(np.sum(np.abs(sv),axis=1),axis=0)
    pd.DataFrame({'feature':names,'mean_abs_shap_sum_over_time':global_mean_abs}).sort_values('mean_abs_shap_sum_over_time',ascending=False).to_csv(out/'global_shap.csv',index=False)
    local=[]
    vals=x.numpy()
    for i,z in enumerate(items):
        signed=np.sum(sv[i],axis=0);meanval=np.mean(vals[i],axis=0)
        for j,f in enumerate(names):local.append({'sample_id':z['sample_id'],'region_group_id':z['group'],'label':int(z['y'].item()),'feature':f,'signed_shap_sum_over_time':float(signed[j]),'standardized_feature_mean':float(meanval[j])})
    pd.DataFrame(local).to_csv(out/'local_shap.csv',index=False)
    rep={'model':a.model,'partition':a.partition,'samples':len(x),'background_samples':len(bg),'sequence_length':int(x.shape[1]),'features':names,'note':'global importance is mean absolute SHAP summed over the 40 temporal positions; local CSV retains signed temporal sums.'}
    (out/'shap_summary.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))

if __name__=='__main__':main()
