from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import all_metrics

ARMS=['real','duplicate','base','hj','pil','hj_pil']
METRICS=['tss','hss','recall','fpr','precision','auroc','auprc','brier','bss','ece10']


def load(root:Path,arm:str):
    d=root/'outputs'/arm
    p=pd.read_csv(d/'test_predictions.csv');m=json.loads((d/'metrics.json').read_text());return p,float(m['validation_threshold']),m


def paired(a,b,ta,tb,nboot,seed):
    keys=['sample_id','region_group_id','y'];z=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(z)!=len(a) or len(z)!=len(b):raise RuntimeError('test prediction sample identities do not match')
    groups=np.asarray(sorted(z.region_group_id.unique()));rng=np.random.default_rng(seed);vals={k:[] for k in METRICS}
    point_a=all_metrics(z.y,z.p_a,ta);point_b=all_metrics(z.y,z.p_b,tb)
    for _ in range(nboot):
        draw=rng.choice(groups,size=len(groups),replace=True);q=pd.concat([z[z.region_group_id.eq(g)] for g in draw],ignore_index=True)
        ma=all_metrics(q.y,q.p_a,ta);mb=all_metrics(q.y,q.p_b,tb)
        for k in METRICS:vals[k].append(ma[k]-mb[k])
    out={}
    for k,v in vals.items():
        x=np.asarray(v,dtype=float);x=x[np.isfinite(x)]
        out[k]={'point_delta':float(point_a[k]-point_b[k]),'median_delta':float(np.median(x)),'lo95':float(np.percentile(x,2.5)),'hi95':float(np.percentile(x,97.5)),'p_two_sided':float(min(1.,2*min(np.mean(x<=0),np.mean(x>=0))))}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--bootstrap',type=int,default=5000);a=ap.parse_args();root=Path(a.root);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    data={};summary={}
    for arm in ARMS:
        p,t,m=load(root,arm);data[arm]=(p,t);summary[arm]={'threshold':t,'test':m['test'],'test_region_bootstrap':m['test_region_bootstrap'],'test_items':m.get('test_items'),'test_groups':m.get('test_groups')}
    base_ids=set(data['real'][0].sample_id)
    for arm in ARMS[1:]:
        if set(data[arm][0].sample_id)!=base_ids:raise RuntimeError(f'{arm}: test sample set differs from real arm')
    comps={}
    seed=260826
    for arm in ARMS[1:]:
        comps[f'{arm}_minus_real']=paired(data[arm][0],data['real'][0],data[arm][1],data['real'][1],a.bootstrap,seed);seed+=1
    for arm in ['base','hj','pil','hj_pil']:
        comps[f'{arm}_minus_duplicate']=paired(data[arm][0],data['duplicate'][0],data[arm][1],data['duplicate'][1],a.bootstrap,seed);seed+=1
    report={'arms':summary,'paired_comparisons':comps,'bootstrap_replicates':a.bootstrap,'interpretation':'delta = first arm minus comparator; lower is better for FPR, Brier, and ECE10'}
    (out/'augmentation_locked_test_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    rows=[]
    for c,metrics in comps.items():
        for k,v in metrics.items():rows.append({'comparison':c,'metric':k,**v})
    pd.DataFrame(rows).to_csv(out/'augmentation_paired_deltas.csv',index=False)
    print(json.dumps(report,indent=2,allow_nan=True),flush=True)

if __name__=='__main__':main()
