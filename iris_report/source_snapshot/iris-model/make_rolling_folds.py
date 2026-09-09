from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--buffer-hours',type=float,default=36.0);a=ap.parse_args()
    receipt=Path(a.evidence_dir)/'data/derived/tai_repair_audit.json'
    if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='PASS':
        raise RuntimeError('Evidence lacks a passing historical TAI repair receipt')
    p=Path(a.evidence_dir)/'data/derived/training_manifest.csv.gz';df=pd.read_csv(p)
    df['t_rec']=pd.to_datetime(df.t_rec,utc=True)
    # Boundary construction is label-blind: only physical-region chronology is used.
    g=df.groupby('region_group_id').agg(start=('t_rec','min'),end=('t_rec','max'),rows=('sample_id','size')).reset_index().sort_values(['start','region_group_id']).reset_index(drop=True)
    n=len(g); specs=[(.40,.55),(.55,.70),(.70,.85),(.85,1.00)];out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);summary=[]
    for fi,(lo,hi) in enumerate(specs,1):
        i0=int(round(n*lo));i1=n if hi>=1 else int(round(n*hi));evalg=g.iloc[i0:i1].copy();eval_start=evalg.start.min();purge=eval_start-pd.Timedelta(hours=a.buffer_hours)
        traing=g[(g.index<i0)&(g.end<purge)].copy()
        tm=df[df.region_group_id.isin(traing.region_group_id)].copy();em=df[df.region_group_id.isin(evalg.region_group_id)].copy()
        if set(tm.region_group_id)&set(em.region_group_id):raise RuntimeError('group leakage')
        tm['fold_role']='train';em['fold_role']='evaluation';z=pd.concat([tm,em],ignore_index=True);z['fold_id']=fi
        path=out/f'fold_{fi}.csv.gz';z.to_csv(path,index=False,compression='gzip')
        summary.append({'fold':fi,'train_groups':int(tm.region_group_id.nunique()),'eval_groups':int(em.region_group_id.nunique()),'train_rows':len(tm),'eval_rows':len(em),'train_positive_groups':int(tm.loc[tm.label_m1plus_24h.eq(1),'region_group_id'].nunique()),'eval_positive_groups':int(em.loc[em.label_m1plus_24h.eq(1),'region_group_id'].nunique()),'eval_start':str(eval_start),'eval_end':str(evalg.end.max()),'buffer_hours':a.buffer_hours})
    blob=''.join(hashlib.sha256((out/f'fold_{i}.csv.gz').read_bytes()).hexdigest() for i in range(1,5));fold_hash=hashlib.sha256(blob.encode()).hexdigest()
    report={'construction':'label-blind group chronology; rolling origin; 36h purge before each evaluation block','fractions':specs,'fold_hash':fold_hash,'folds':summary}
    (out/'rolling_folds_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
