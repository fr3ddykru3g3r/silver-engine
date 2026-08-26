from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import build_records, MagnetogramDataset, cache_records

FEATURES10=['R_VALUE','AREA_ACR','TOTUSJZ','TOTUSJH','TOTPOT','ABSNJZH','SAVNCPP','USFLUX','MEANPOT','SHRGT45']
FEATURES2=['R_VALUE','AREA_ACR']


def temporal_group_cap(df: pd.DataFrame, per_group: int, pos_cap: int, seed: int = 2026) -> pd.DataFrame:
    """Cap each physical region while preserving temporal spread and positives.

    The purpose is not to cherry-pick flaring windows; it prevents one long-lived
    active region from dominating simply because it contributes many hourly rows.
    """
    if per_group <= 0:
        return df.copy().reset_index(drop=True)
    rng=np.random.default_rng(seed); pieces=[]
    work=df.copy()
    tcol='t_rec' if 't_rec' in work.columns else ('end_time' if 'end_time' in work.columns else None)
    if tcol: work['_t']=pd.to_datetime(work[tcol],utc=True,errors='coerce')
    for g,z in work.groupby('region_group_id',sort=True):
        z=z.sort_values('_t' if '_t' in z else z.index.name or z.columns[0])
        pos=z[z.label_m1plus_24h.eq(1)]
        neg=z[z.label_m1plus_24h.eq(0)]
        want_pos=min(len(pos), max(0,min(pos_cap,per_group)))
        def spread(x,n):
            if n<=0:return x.iloc[0:0]
            if len(x)<=n:return x
            ids=np.unique(np.round(np.linspace(0,len(x)-1,n)).astype(int))
            if len(ids)<n:
                extra=[i for i in range(len(x)) if i not in set(ids)][:n-len(ids)]
                ids=np.r_[ids,extra]
            return x.iloc[ids[:n]]
        p=spread(pos,want_pos)
        n=spread(neg,min(len(neg),per_group-len(p)))
        q=pd.concat([p,n])
        if len(q)<min(per_group,len(z)):
            rest=z[~z.index.isin(q.index)]
            q=pd.concat([q,spread(rest,min(per_group-len(q),len(rest)))])
        pieces.append(q)
    out=pd.concat(pieces,ignore_index=True) if pieces else work.iloc[0:0].copy()
    return out.drop(columns=['_t'],errors='ignore').sample(frac=1,random_state=seed).reset_index(drop=True)


def build_point_image_records(evidence_dir: str|Path, partition: str, per_group: int=0, pos_cap: int=4,
                              seed: int=2026) -> pd.DataFrame:
    x=build_records(evidence_dir,partition)
    return temporal_group_cap(x,per_group,pos_cap,seed)


class FeatureSequenceDataset(Dataset):
    def __init__(self, evidence_dir: str|Path, partition: str, n_features: int=10,
                 per_group: int=0, pos_cap: int=4, seed: int=2026):
        d=Path(evidence_dir)/'data'/'derived'
        idx=pd.read_csv(d/'cdr_sequence_index.csv.gz',low_memory=False)
        idx=idx[idx.partition.eq(partition)].copy()
        idx=temporal_group_cap(idx,per_group,pos_cap,seed)
        feat=pd.read_csv(d/'cdr_feature_metadata.csv.gz',low_memory=False)
        stats=json.loads((d/'cdr_feature_train_stats.json').read_text())
        names=FEATURES10 if int(n_features)==10 else FEATURES2
        self.names=names; self.seq_len=int(stats['seq_len']); self.rows=idx.reset_index(drop=True)
        # Parse JSOC T_REC exactly as in the rest of the pipeline.
        t=feat.T_REC.astype(str).str.replace('_TAI','',regex=False)
        t=t.str.replace(r'^(\d{4})\.(\d{2})\.(\d{2})_',r'\1-\2-\3T',regex=True)
        feat['time']=pd.to_datetime(t,utc=True,errors='coerce').dt.floor('h')
        feat['harpnum']=pd.to_numeric(feat.HARPNUM,errors='coerce').astype('Int64')
        feat=feat[feat.time.notna() & feat.harpnum.notna()].copy()
        for f in names:feat[f]=pd.to_numeric(feat[f],errors='coerce')
        st=stats['stats']; self.median=np.array([st[f]['median'] for f in names],dtype=np.float32)
        self.mean=np.array([st[f]['mean'] for f in names],dtype=np.float32)
        self.std=np.array([st[f]['std'] for f in names],dtype=np.float32)
        self.lookup={}
        for _,r in feat.iterrows():
            self.lookup[(int(r.harpnum),r.time)]=np.array([r[f] for f in names],dtype=np.float32)

    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]; h=int(r.harpnum); end=pd.Timestamp(r.end_time)
        if end.tzinfo is None:end=end.tz_localize('UTC')
        vals=[]
        for j in range(self.seq_len):
            t=end-pd.Timedelta(hours=self.seq_len-1-j)
            x=self.lookup[(h,t)].copy(); bad=~np.isfinite(x); x[bad]=self.median[bad]
            vals.append((x-self.mean)/self.std)
        return {'x':torch.from_numpy(np.stack(vals).astype(np.float32)),
                'y':torch.tensor(float(r.label_m1plus_24h),dtype=torch.float32),
                'group':str(r.region_group_id),'sample_id':str(r.sample_id)}


def feature_index(evidence_dir: str|Path, partition: str) -> pd.DataFrame:
    d=Path(evidence_dir)/'data'/'derived'
    x=pd.read_csv(d/'cdr_sequence_index.csv.gz',low_memory=False)
    return x[x.partition.eq(partition)].copy().reset_index(drop=True)


def build_image_sequence_endpoints(evidence_dir: str|Path, partition: str, seq_len: int=24,
                                   per_group: int=1, pos_cap: int=1, seed: int=2026):
    """Return endpoint rows and sample-id lists for clean hourly image sequences.

    Only primary rows already passing the integrity/quality/CMD gate are used in
    these sequences. This is stricter than the feature-sequence history rule and
    avoids importing lower-quality magnetograms into the image comparator.
    """
    df=build_records(evidence_dir,partition)
    df['time']=pd.to_datetime(df.t_rec,utc=True,errors='coerce').dt.floor('h')
    by={(int(r.harpnum),r.time):str(r.sample_id) for _,r in df.iterrows()}
    rows=[]
    for _,r in df.iterrows():
        h=int(r.harpnum);end=r.time
        ids=[by.get((h,end-pd.Timedelta(hours=seq_len-1-j))) for j in range(seq_len)]
        if any(x is None for x in ids):continue
        z=r.to_dict();z['sequence_sample_ids']=';'.join(ids);z['end_time']=end.isoformat();rows.append(z)
    out=pd.DataFrame(rows)
    return temporal_group_cap(out,per_group,pos_cap,seed)


class CachedImageSequenceDataset(Dataset):
    def __init__(self, cached_records: pd.DataFrame, endpoints: pd.DataFrame):
        self.end=endpoints.reset_index(drop=True); self.records=cached_records.reset_index(drop=True)
        self.pos={str(r.sample_id):i for i,r in self.records.iterrows()}
        self.base=MagnetogramDataset(self.records)
    def __len__(self):return len(self.end)
    def __getitem__(self,i):
        r=self.end.iloc[i]; ids=str(r.sequence_sample_ids).split(';'); frames=[]
        for sid in ids:
            frames.append(self.base[self.pos[sid]]['x'])
        return {'x':torch.stack(frames,0),'y':torch.tensor(float(r.label_m1plus_24h),dtype=torch.float32),
                'group':str(r.region_group_id),'sample_id':str(r.sample_id)}


def cache_image_sequences(evidence_dir: str|Path, partition: str, cache_dir: str|Path,
                          seq_len: int=24, per_group: int=1, pos_cap: int=1, seed: int=2026,
                          workers: int=12):
    end=build_image_sequence_endpoints(evidence_dir,partition,seq_len,per_group,pos_cap,seed)
    allrec=build_records(evidence_dir,partition)
    ids=set()
    for s in end.sequence_sample_ids.astype(str):ids.update(s.split(';'))
    need=allrec[allrec.sample_id.astype(str).isin(ids)].copy()
    need=cache_records(need,cache_dir,workers)
    return need,end
