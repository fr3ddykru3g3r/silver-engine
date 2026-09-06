from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def eligible_rows(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path)
    x['t_rec_dt']=pd.to_datetime(x.t_rec,utc=True,errors='raise')
    q=x.quality.astype(str).str.lower().eq('0x00000000')
    clean=x.label_integrity_status.astype(str).eq('RESOLVED_OR_CLEAN')
    y=x.label_m1plus_24h.isin([0,1])
    cmd=x.cmd_deg.abs().le(30.0)
    z=x[q&clean&y&cmd].copy()
    z['label_m1plus_24h']=z.label_m1plus_24h.astype(int)
    return z


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--initial-train-fraction',type=float,default=0.60)
    ap.add_argument('--outer-folds',type=int,default=4)
    ap.add_argument('--buffer-hours',type=float,default=36.0)
    args=ap.parse_args()

    if not (0.4 <= args.initial_train_fraction < 0.9):
        raise ValueError('initial-train-fraction outside predeclared sensible range')
    if args.outer_folds < 2:
        raise ValueError('Need at least two outer folds')

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    src=Path(args.evidence_dir)/'data/derived/training_manifest.csv.gz'
    receipt=src.with_name('tai_repair_audit.json')
    if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='PASS':
        raise RuntimeError('Evidence lacks a passing historical TAI repair receipt')
    x=eligible_rows(src)

    g=(x.groupby('region_group_id',sort=False)
         .agg(group_start=('t_rec_dt','min'),group_end=('t_rec_dt','max'),rows=('sample_id','size'))
         .reset_index()
         .sort_values(['group_start','region_group_id'])
         .reset_index(drop=True))

    n=len(g); initial=int(np.floor(n*args.initial_train_fraction))
    remaining=n-initial
    edges=np.linspace(initial,n,args.outer_folds+1).round().astype(int)
    # Force exact monotonic integer edges with all groups assigned once to an outer block.
    edges[0]=initial; edges[-1]=n
    if np.any(np.diff(edges)<=0): raise RuntimeError('Invalid fold edges')

    buffer=pd.Timedelta(hours=args.buffer_hours)
    summaries=[]; all_eval=set()
    for k in range(args.outer_folds):
        a,b=int(edges[k]),int(edges[k+1])
        eval_groups=g.iloc[a:b].copy()
        cut=eval_groups.group_start.min()
        candidate_train=g.iloc[:a].copy()
        # Whole connected regions only. Any earlier-starting group extending into the
        # boundary buffer is excluded from that fold rather than truncated.
        train_groups=candidate_train[candidate_train.group_end < cut-buffer].copy()
        train_ids=set(train_groups.region_group_id.astype(str))
        eval_ids=set(eval_groups.region_group_id.astype(str))
        if train_ids & eval_ids: raise RuntimeError('region_group_id leakage')
        if all_eval & eval_ids: raise RuntimeError('outer evaluation group reused')
        all_eval |= eval_ids

        tr=x[x.region_group_id.astype(str).isin(train_ids)].drop(columns=['t_rec_dt']).copy()
        ev=x[x.region_group_id.astype(str).isin(eval_ids)].drop(columns=['t_rec_dt']).copy()
        fd=out/f'fold_{k+1}'; fd.mkdir(exist_ok=True)
        tp=fd/'train.csv.gz'; ep=fd/'evaluation.csv.gz'
        tr.to_csv(tp,index=False,compression='gzip'); ev.to_csv(ep,index=False,compression='gzip')

        s={
            'fold':k+1,
            'selection_rule':'connected groups sorted by first eligible observation; first 60% initial history; remaining groups split into four equal-count chronological blocks; no outcome/performance metric defines boundaries',
            'cut_time_utc':str(cut),
            'buffer_hours':args.buffer_hours,
            'candidate_train_groups_before_buffer':len(candidate_train),
            'excluded_boundary_groups':len(candidate_train)-len(train_groups),
            'train_groups':len(train_groups),'evaluation_groups':len(eval_groups),
            'train_rows':len(tr),'evaluation_rows':len(ev),
            'train_positive_rows':int(tr.label_m1plus_24h.sum()),'evaluation_positive_rows':int(ev.label_m1plus_24h.sum()),
            'train_positive_groups':int(tr.loc[tr.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
            'evaluation_positive_groups':int(ev.loc[ev.label_m1plus_24h.eq(1),'region_group_id'].nunique()),
            'train_sha256':sha256_file(tp),'evaluation_sha256':sha256_file(ep),
        }
        (fd/'fold_summary.json').write_text(json.dumps(s,indent=2)+'\n')
        summaries.append(s)

    report={
        'protocol':'V2 frozen rolling-origin outer evaluation',
        'source_manifest_sha256':sha256_file(src),
        'eligible_rows':len(x),'eligible_connected_groups':n,
        'initial_train_fraction':args.initial_train_fraction,'outer_folds':args.outer_folds,
        'buffer_hours':args.buffer_hours,
        'boundary_definition_uses_labels':False,
        'note':'Positive counts are reported only as a post-construction power audit; they do not define boundaries.',
        'folds':summaries,
    }
    (out/'rolling_origin_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
