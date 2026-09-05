from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--buffer-hours',type=float,default=36.0)
    ap.add_argument('--train-fracs',default='0.55,0.65,0.75,0.85')
    ap.add_argument('--eval-width-frac',type=float,default=0.10)
    a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.manifest)
    df['t_rec']=pd.to_datetime(df.t_rec,utc=True)
    req={'sample_id','region_group_id','t_rec'}
    if not req.issubset(df.columns): raise RuntimeError(f'missing {req-set(df.columns)}')
    g=df.groupby('region_group_id').t_rec.agg(['min','max']).sort_values('min').reset_index()
    n=len(g); buf=pd.Timedelta(hours=a.buffer_hours)
    folds=[]
    for i,frac in enumerate([float(x) for x in a.train_fracs.split(',')]):
        cut_idx=max(1,min(n-2,int(np.floor(frac*n))))
        eval_end_idx=max(cut_idx+1,min(n,int(np.floor((frac+a.eval_width_frac)*n))))
        boundary=g.iloc[cut_idx]['min']; eval_end=g.iloc[eval_end_idx-1]['min'] if eval_end_idx<n else g.iloc[-1]['max']+pd.Timedelta(seconds=1)
        train_groups=set(g.loc[g['max'] < boundary-buf,'region_group_id'].astype(str))
        eval_groups=set(g.loc[(g['min'] > boundary+buf) & (g['min'] < eval_end),'region_group_id'].astype(str))
        tr=df[df.region_group_id.astype(str).isin(train_groups)].copy(); ev=df[df.region_group_id.astype(str).isin(eval_groups)].copy()
        if not tr.empty and not ev.empty and not (tr.t_rec.max()+buf < ev.t_rec.min()): raise RuntimeError('strict temporal buffer failed')
        if set(tr.region_group_id.astype(str)) & set(ev.region_group_id.astype(str)): raise RuntimeError('group overlap')
        fp=out/f'fold_{i+1}_train.csv.gz'; ep=out/f'fold_{i+1}_eval.csv.gz'
        tr.to_csv(fp,index=False,compression='gzip'); ev.to_csv(ep,index=False,compression='gzip')
        folds.append({'fold':i+1,'fraction_boundary':frac,'boundary_utc':str(boundary),'buffer_hours':a.buffer_hours,'train_rows':len(tr),'train_groups':tr.region_group_id.nunique(),'eval_rows':len(ev),'eval_groups':ev.region_group_id.nunique(),'train_positive_rows':int(tr.label_m1plus_24h.sum()) if 'label_m1plus_24h' in tr else None,'eval_positive_rows':int(ev.label_m1plus_24h.sum()) if 'label_m1plus_24h' in ev else None,'train_sha256':sha256_file(fp),'eval_sha256':sha256_file(ep)})
    report={'selection_rule':'boundaries chosen from region chronology only; labels are reported after fold construction and never used to set boundaries','buffer_hours':a.buffer_hours,'folds':folds}
    (out/'rolling_folds.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
