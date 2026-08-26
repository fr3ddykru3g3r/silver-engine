#!/usr/bin/env python3
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'
FEATURES10=['R_VALUE','AREA_ACR','TOTUSJZ','TOTUSJH','TOTPOT','ABSNJZH','SAVNCPP','USFLUX','MEANPOT','SHRGT45']
FEATURES2=['R_VALUE','AREA_ACR']
SEQ_LEN=40
STEP_MINUTES=36


def parse_trec(s: pd.Series) -> pd.Series:
    x=s.astype(str).str.replace('_TAI','',regex=False)
    x=x.str.replace(r'^(\d{4})\.(\d{2})\.(\d{2})_',r'\1-\2-\3T',regex=True)
    # Keep native 12-minute timestamps exactly; do not floor to the hour.
    return pd.to_datetime(x,utc=True,errors='coerce')


def quality_ok(v):
    try:return int(str(v),0)==0
    except:
        try:return int(float(v))==0
        except:return False


def main():
    man=pd.read_csv(DER/'training_manifest.csv.gz',low_memory=False)
    feat=pd.read_csv(DER/'cdr_feature_metadata.csv.gz',low_memory=False)
    man=man[man.partition.isin(['train','validation','test']) & man.label_m1plus_24h.notna()].copy()
    man['time']=pd.to_datetime(man.t_rec,utc=True,errors='coerce')
    man['harpnum']=pd.to_numeric(man.harpnum,errors='coerce').astype('Int64')
    feat['time']=parse_trec(feat.T_REC)
    feat['harpnum']=pd.to_numeric(feat.HARPNUM,errors='coerce').astype('Int64')
    feat=feat[feat.time.notna() & feat.harpnum.notna()].copy();feat['harpnum']=feat.harpnum.astype(int)
    for f in FEATURES10:feat[f]=pd.to_numeric(feat[f],errors='coerce')
    feat['quality_ok']=feat.QUALITY.map(quality_ok)
    feat['cmd']=pd.to_numeric(feat.LON_FWT,errors='coerce')
    feat['history_ok']=feat.quality_ok & feat.cmd.abs().le(45)
    feat=feat.sort_values(['harpnum','time']).drop_duplicates(['harpnum','time'])

    fmap={(int(r.harpnum),r.time):r for _,r in feat.iterrows()}
    seq_rows=[]
    step=pd.Timedelta(minutes=STEP_MINUTES)
    for _,r in man.sort_values(['harpnum','time']).iterrows():
        if pd.isna(r.time) or pd.isna(r.harpnum):continue
        h=int(r.harpnum); end=r.time
        # 40 actual observations separated by 36 minutes span 23 h 24 min,
        # matching the paper's 40-sample / 36-minute design over a 24 h state.
        times=[end-step*(SEQ_LEN-1-i) for i in range(SEQ_LEN)]
        states=[fmap.get((h,t)) for t in times]
        if any(s is None for s in states):continue
        if not all(bool(s.history_ok) for s in states):continue
        urls=[str(s.segment_magnetogram) for s in states]
        if any((not u or u in {'nan','None'}) for u in urls):continue
        seq_rows.append({'sample_id':r.sample_id,'harpnum':h,'region_group_id':r.region_group_id,
                         'partition':r.partition,'label_m1plus_24h':int(r.label_m1plus_24h),
                         'end_time':end.isoformat(),'start_time':times[0].isoformat(),
                         'sequence_length':SEQ_LEN,'sequence_step_minutes':STEP_MINUTES,
                         'sequence_times':'|'.join(t.isoformat() for t in times),
                         'sequence_magnetogram_urls':'|'.join(urls)})
    idx=pd.DataFrame(seq_rows)
    if idx.empty:raise SystemExit('No valid CDR sequences')
    idx.to_csv(DER/'cdr_sequence_index.csv.gz',index=False,compression='gzip')

    train_ids=set(idx.loc[idx.partition.eq('train'),'sample_id'])
    train_end=man[man.sample_id.isin(train_ids)][['sample_id','harpnum','time']]
    train_keys=set()
    for _,r in train_end.iterrows():
        for j in range(SEQ_LEN):train_keys.add((int(r.harpnum),r.time-step*j))
    keymask=[(int(r.harpnum),r.time) in train_keys for _,r in feat.iterrows()]
    train_feat=feat[keymask].copy()
    stats={}
    for f in FEATURES10:
        z=train_feat[f].replace([np.inf,-np.inf],np.nan)
        med=float(z.median()) if z.notna().any() else 0.0
        zi=z.fillna(med)
        mean=float(zi.mean());std=float(zi.std(ddof=0))
        if not math.isfinite(std) or std<1e-8:std=1.0
        stats[f]={'median':med,'mean':mean,'std':std}
    (DER/'cdr_feature_train_stats.json').write_text(json.dumps({'seq_len':SEQ_LEN,'step_minutes':STEP_MINUTES,'features10':FEATURES10,'features2':FEATURES2,'stats':stats},indent=2)+'\n')
    rep={'sequence_length':SEQ_LEN,'step_minutes':STEP_MINUTES,'history_span_minutes':(SEQ_LEN-1)*STEP_MINUTES,
         'rows':len(idx),'groups':int(idx.region_group_id.nunique()),'partitions':[]}
    for p in ['train','validation','test']:
        x=idx[idx.partition.eq(p)];pos=x[x.label_m1plus_24h.eq(1)]
        rep['partitions'].append({'partition':p,'sequences':len(x),'positive_sequences':len(pos),
                                  'groups':int(x.region_group_id.nunique()),
                                  'positive_groups':int(pos.region_group_id.nunique())})
    (DER/'cdr_sequence_audit.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps(rep,indent=2),flush=True)

if __name__=='__main__':main()
