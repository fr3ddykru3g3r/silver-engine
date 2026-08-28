from __future__ import annotations

"""Build one leakage-safe nested evidence bundle from a frozen rolling-origin fold.

Outer train/evaluation connected-region blocks are immutable inputs. The outer
training history is split again by connected region and time into inner train and
validation sets. A 36 h buffer is enforced by dropping whole boundary-crossing
regions. The outer evaluation block is assigned partition=test only after all
inner/model-selection choices have been defined.

No performance metric defines any boundary.
"""

import argparse, json, shutil
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True)
    ap.add_argument('--outer-fold-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--inner-train-fraction',type=float,default=0.80)
    ap.add_argument('--buffer-hours',type=float,default=36.0)
    a=ap.parse_args()
    if not 0.6 <= a.inner_train_fraction < 0.95: raise ValueError('inner-train-fraction')

    src=Path(a.evidence_dir); fd=Path(a.outer_fold_dir); out=Path(a.out_dir)
    shutil.copytree(src,out,dirs_exist_ok=True)
    tr=pd.read_csv(fd/'train.csv.gz',low_memory=False); te=pd.read_csv(fd/'evaluation.csv.gz',low_memory=False)
    tr['dt']=pd.to_datetime(tr.t_rec,utc=True,errors='raise'); te['dt']=pd.to_datetime(te.t_rec,utc=True,errors='raise')
    g=(tr.groupby('region_group_id').agg(start=('dt','min'),end=('dt','max')).reset_index().sort_values(['start','region_group_id']).reset_index(drop=True))
    cut_idx=max(1,min(len(g)-1,int(len(g)*a.inner_train_fraction)))
    val_groups=g.iloc[cut_idx:].copy(); cut=val_groups.start.min(); buffer=pd.Timedelta(hours=a.buffer_hours)
    train_groups=g.iloc[:cut_idx]; train_groups=train_groups[train_groups.end < cut-buffer]
    train_ids=set(train_groups.region_group_id.astype(str)); val_ids=set(val_groups.region_group_id.astype(str)); test_ids=set(te.region_group_id.astype(str))
    if train_ids&val_ids or train_ids&test_ids or val_ids&test_ids: raise RuntimeError('connected-region leakage')
    itr=tr[tr.region_group_id.astype(str).isin(train_ids)].drop(columns='dt').copy(); iva=tr[tr.region_group_id.astype(str).isin(val_ids)].drop(columns='dt').copy(); ote=te.drop(columns='dt').copy()
    itr['partition']='train'; iva['partition']='validation'; ote['partition']='test'
    man=pd.concat([itr,iva,ote],ignore_index=True)
    p=out/'data/derived/training_manifest.csv.gz'; man.to_csv(p,index=False,compression='gzip')
    rep={'protocol':'nested connected-region chronological split inside frozen rolling-origin outer fold','boundary_uses_outcomes':False,'inner_train_fraction':a.inner_train_fraction,'buffer_hours':a.buffer_hours,'inner_train_groups':len(train_ids),'inner_validation_groups':len(val_ids),'outer_test_groups':len(test_ids),'inner_train_rows':len(itr),'inner_validation_rows':len(iva),'outer_test_rows':len(ote),'inner_train_positive_groups':int(itr.loc[itr.label_m1plus_24h.eq(1),'region_group_id'].nunique()),'inner_validation_positive_groups':int(iva.loc[iva.label_m1plus_24h.eq(1),'region_group_id'].nunique()),'outer_test_positive_groups':int(ote.loc[ote.label_m1plus_24h.eq(1),'region_group_id'].nunique()),'outer_test_not_used_for_selection':True}
    (out/'nested_split_audit.json').write_text(json.dumps(rep,indent=2)+'\n'); print(json.dumps(rep,indent=2))

if __name__=='__main__': main()
