from __future__ import annotations

import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()


def group_table(x: pd.DataFrame) -> pd.DataFrame:
    z=x.copy(); z['dt']=pd.to_datetime(z.t_rec,utc=True,errors='raise')
    return (z.groupby('region_group_id',sort=False)
             .agg(group_start=('dt','min'),group_end=('dt','max'))
             .reset_index().sort_values(['group_start','region_group_id']).reset_index(drop=True))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outer-fold-root',required=True)
    ap.add_argument('--base-evidence-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--inner-validation-fraction',type=float,default=0.15)
    ap.add_argument('--buffer-hours',type=float,default=36.0)
    args=ap.parse_args()
    root=Path(args.outer_fold_root); base=Path(args.base_evidence_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    if not (0.10 <= args.inner_validation_fraction <= 0.25): raise ValueError('inner validation fraction outside frozen range')
    meta=base/'data/derived/sharp_metadata.csv.gz'
    if not meta.exists(): raise FileNotFoundError(meta)
    summaries=[]
    for fd in sorted(root.glob('fold_*')):
        outer_train=pd.read_csv(fd/'train.csv.gz',low_memory=False)
        outer_eval=pd.read_csv(fd/'evaluation.csv.gz',low_memory=False)
        g=group_table(outer_train); n=len(g)
        nv=max(1,int(round(n*args.inner_validation_fraction))); split=n-nv
        valg=g.iloc[split:].copy(); cut=valg.group_start.min(); buf=pd.Timedelta(hours=args.buffer_hours)
        trg=g.iloc[:split].copy(); trg=trg[trg.group_end < cut-buf].copy()
        tr_ids=set(trg.region_group_id.astype(str)); va_ids=set(valg.region_group_id.astype(str)); te_ids=set(outer_eval.region_group_id.astype(str))
        if tr_ids&va_ids or tr_ids&te_ids or va_ids&te_ids: raise RuntimeError(f'{fd.name}: group leakage')
        tr=outer_train[outer_train.region_group_id.astype(str).isin(tr_ids)].copy(); tr['partition']='train'
        va=outer_train[outer_train.region_group_id.astype(str).isin(va_ids)].copy(); va['partition']='validation'
        te=outer_eval.copy(); te['partition']='test'
        man=pd.concat([tr,va,te],ignore_index=True).sort_values(['t_rec','region_group_id','sample_id']).reset_index(drop=True)
        dest=out/fd.name/'data/derived'; dest.mkdir(parents=True,exist_ok=True)
        mp=dest/'training_manifest.csv.gz'; man.to_csv(mp,index=False,compression='gzip')
        shutil.copy2(meta,dest/'sharp_metadata.csv.gz')
        # Keep only the two files needed by build_records; this prevents accidental access to unrelated original partitions.
        s={
          'fold':int(fd.name.split('_')[-1]),
          'inner_validation_fraction':args.inner_validation_fraction,'buffer_hours':args.buffer_hours,
          'inner_validation_cut_time_utc':str(cut),
          'train_groups':len(tr_ids),'validation_groups':len(va_ids),'test_groups':len(te_ids),
          'train_rows':len(tr),'validation_rows':len(va),'test_rows':len(te),
          'train_positive_groups':int(tr.loc[tr.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
          'validation_positive_groups':int(va.loc[va.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
          'test_positive_groups':int(te.loc[te.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
          'manifest_sha256':sha256_file(mp),'sharp_metadata_sha256':sha256_file(dest/'sharp_metadata.csv.gz'),
          'test_equals_frozen_outer_evaluation':set(te.sample_id.astype(str))==set(outer_eval.sample_id.astype(str)),
        }
        (out/fd.name/'nested_split_summary.json').write_text(json.dumps(s,indent=2)+'\n'); summaries.append(s)
    report={'protocol':'V2 nested chronological folds: generator/forecaster train on inner train; threshold/checkpoint selection on real-only inner validation; final evaluation on frozen outer block',
            'inner_boundary_uses_labels':False,'folds':summaries}
    (out/'nested_folds_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
