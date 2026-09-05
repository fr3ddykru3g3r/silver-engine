from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import all_metrics

PRIMARY='tss'
DEFAULT_ARMS=['real','rw','duplicate','base','hj','pil','hj_pil','pil_blur','geometry_flip','block_shuffle']


def load_arm(d: Path):
    p=pd.read_csv(d/'test_predictions.csv')
    m=json.loads((d/'metrics.json').read_text())
    thr=float(m['validation_threshold'])
    return p,thr,m


def paired_delta(a,b,ta,tb,metric='tss'):
    keys=['sample_id','region_group_id','y']
    z=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(z)!=len(a) or len(z)!=len(b): raise RuntimeError('Prediction identities differ')
    ma=all_metrics(z.y,z.p_a,ta); mb=all_metrics(z.y,z.p_b,tb)
    return float(ma[metric]-mb[metric]),z


def fold_bootstrap_delta(z,ta,tb,metric,rng):
    groups=np.asarray(sorted(z.region_group_id.astype(str).unique()))
    draw=rng.choice(groups,size=len(groups),replace=True)
    q=pd.concat([z[z.region_group_id.astype(str).eq(g)] for g in draw],ignore_index=True)
    a=all_metrics(q.y,q.p_a,ta); b=all_metrics(q.y,q.p_b,tb)
    return float(a[metric]-b[metric])


def stratified_bootstrap(folds,arm,ref,metric,nboot,seed):
    rng=np.random.default_rng(seed); fold_data=[]; points=[]
    for f,data in folds.items():
        a,ta,_=data[arm]; b,tb,_=data[ref]
        d,z=paired_delta(a,b,ta,tb,metric); points.append(d); fold_data.append((f,z,ta,tb,d))
    boots=[]
    for _ in range(nboot):
        vals=[fold_bootstrap_delta(z,ta,tb,metric,rng) for _,z,ta,tb,_ in fold_data]
        boots.append(float(np.mean(vals)))
    x=np.asarray(boots,float)
    return {
      'metric':metric,'arm':arm,'reference':ref,
      'equal_fold_mean_point_delta':float(np.mean(points)),
      'per_fold_point_delta':{str(f):float(d) for f,_,_,_,d in fold_data},
      'bootstrap_median':float(np.median(x)),'lo95':float(np.percentile(x,2.5)),'hi95':float(np.percentile(x,97.5)),
      'p_two_sided':float(min(1.0,2*min(np.mean(x<=0),np.mean(x>=0)))),
      'bootstrap_replicates':nboot,
      'resampling_unit':'connected region within each chronological fold; folds weighted equally',
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--arms',nargs='*',default=DEFAULT_ARMS); ap.add_argument('--bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=260826)
    args=ap.parse_args(); root=Path(args.root); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    fold_dirs=sorted([p for p in root.glob('fold_*') if p.is_dir()])
    if len(fold_dirs)<2: raise RuntimeError('Need >=2 chronological folds')
    folds={}
    for fd in fold_dirs:
        f=int(fd.name.split('_')[-1]); folds[f]={}
        for arm in args.arms:
            d=fd/arm
            if not (d/'test_predictions.csv').exists(): continue
            folds[f][arm]=load_arm(d)
        if 'duplicate' not in folds[f] or 'real' not in folds[f]: raise RuntimeError(f'{fd}: real/duplicate missing')
        base_ids=set(folds[f]['real'][0].sample_id.astype(str))
        for arm,(p,_,_) in folds[f].items():
            if set(p.sample_id.astype(str)) != base_ids: raise RuntimeError(f'{fd}/{arm}: outer evaluation identities differ')

    comparisons=[]
    for arm in args.arms:
        if arm in ('real','rw','duplicate'): continue
        if all(arm in d for d in folds.values()): comparisons.append((arm,'duplicate'))
    for arm in ('duplicate','rw','base','hj','pil','hj_pil'):
        if all(arm in d for d in folds.values()): comparisons.append((arm,'real'))
    for pair in [('pil','pil_blur'),('hj','geometry_flip'),('base','block_shuffle')]:
        if all(pair[0] in d and pair[1] in d for d in folds.values()): comparisons.append(pair)

    report={'protocol':'V2 frozen rolling-origin paired analysis','primary_metric':PRIMARY,'folds':sorted(folds),
            'comparisons':{},'interpretation':'delta = first arm minus reference; positive TSS/HSS/AUROC/AUPRC is better; negative Brier/FPR is better'}
    metrics=['tss','hss','recall','fpr','precision','auroc','auprc','brier','bss']
    s=args.seed
    for arm,ref in comparisons:
        key=f'{arm}_minus_{ref}'; report['comparisons'][key]={}
        for metric in metrics:
            report['comparisons'][key][metric]=stratified_bootstrap(folds,arm,ref,metric,args.bootstrap,s); s+=1
    (out/'rolling_transfer_comparisons.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    rows=[]
    for c,mm in report['comparisons'].items():
        for metric,r in mm.items(): rows.append({'comparison':c,'metric':metric,**{k:v for k,v in r.items() if k not in ('arm','reference','metric','per_fold_point_delta')},'per_fold_point_delta_json':json.dumps(r['per_fold_point_delta'])})
    pd.DataFrame(rows).to_csv(out/'rolling_transfer_comparisons.csv',index=False)
    print(json.dumps(report,indent=2,allow_nan=True))

if __name__=='__main__': main()
