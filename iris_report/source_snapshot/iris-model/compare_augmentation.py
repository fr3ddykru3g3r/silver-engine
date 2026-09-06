from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import all_metrics

ARMS=['real','rw','duplicate','base','hj','pil','hj_pil']


def load_arm(root:Path,arm:str):
    hits=list(root.rglob(f'{arm}/validation_predictions.csv'))
    if not hits: hits=[p for p in root.rglob('validation_predictions.csv') if arm in str(p)]
    if not hits: raise FileNotFoundError(f'validation predictions for {arm}')
    p=hits[0];df=pd.read_csv(p);m=json.loads((p.parent/'metrics.json').read_text());return df,m


def paired_bootstrap(base,other,base_thr,other_thr,n_boot,seed):
    a=base[['sample_id','region_group_id','y','p']].rename(columns={'p':'pa'});b=other[['sample_id','region_group_id','y','p']].rename(columns={'p':'pb'})
    z=a.merge(b,on=['sample_id','region_group_id','y'],how='inner',validate='one_to_one')
    if len(z)!=len(a) or len(z)!=len(b):raise RuntimeError('arms do not share identical validation samples')
    rng=np.random.default_rng(seed);groups=np.asarray(sorted(z.region_group_id.unique()));vals=[]
    for _ in range(n_boot):
        draw=rng.choice(groups,size=len(groups),replace=True);pieces=[]
        for j,g in enumerate(draw):
            q=z[z.region_group_id.eq(g)].copy();q['_draw']=j;pieces.append(q)
        q=pd.concat(pieces,ignore_index=True);ma=all_metrics(q.y,q.pa,base_thr);mb=all_metrics(q.y,q.pb,other_thr)
        vals.append({k:mb[k]-ma[k] for k in ['tss','hss','auroc','auprc','brier','bss']})
    out={}
    for k in vals[0]:
        v=np.asarray([x[k] for x in vals],dtype=float);med=float(np.nanmedian(v));lo=float(np.nanpercentile(v,2.5));hi=float(np.nanpercentile(v,97.5));p2=float(min(1.,2*min(np.nanmean(v<=0),np.nanmean(v>=0))))
        out[k]={'median_delta':med,'lo95':lo,'hi95':hi,'bootstrap_two_sided_p':p2}
    return out,len(z),len(groups)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--n-boot',type=int,default=5000);ap.add_argument('--seed',type=int,default=2026);args=ap.parse_args()
    root=Path(args.root);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);data={};metrics={}
    for a in ARMS:data[a],metrics[a]=load_arm(root,a)
    rows=[];details={}
    for a in ARMS:
        v=metrics[a]['validation'];rows.append({'arm':a,'threshold':metrics[a]['validation_threshold'],**{f'val_{k}':v.get(k) for k in ['tss','hss','auroc','auprc','brier','bss','ece10']}})
    pd.DataFrame(rows).sort_values(['val_tss','val_auprc'],ascending=False).to_csv(out/'validation_arm_ranking.csv',index=False)
    for ref in ['real','duplicate']:
        for a in [x for x in ARMS if x!=ref]:
            d,n,ng=paired_bootstrap(data[ref],data[a],float(metrics[ref]['validation_threshold']),float(metrics[a]['validation_threshold']),args.n_boot,args.seed)
            details[f'{a}_minus_{ref}']={'rows':n,'groups':ng,'metrics':d}
    (out/'paired_region_bootstrap_differences.json').write_text(json.dumps(details,indent=2,allow_nan=True)+'\n')
    print(json.dumps({'arms':rows,'paired_comparisons':details},indent=2,allow_nan=True))

if __name__=='__main__':main()
