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


def _parse_trec(s: pd.Series) -> pd.Series:
    x=s.astype(str).str.replace('_TAI','',regex=False)
    x=x.str.replace(r'^(\d{4})\.(\d{2})\.(\d{2})_',r'\1-\2-\3T',regex=True)
    return pd.to_datetime(x,utc=True,errors='coerce')


def temporal_group_cap(df: pd.DataFrame, per_group: int, pos_cap: int, seed: int = 2026) -> pd.DataFrame:
    """Cap each connected physical region while preserving temporal spread."""
    if per_group <= 0:return df.copy().reset_index(drop=True)
    pieces=[];work=df.copy()
    tcol='t_rec' if 't_rec' in work.columns else ('end_time' if 'end_time' in work.columns else None)
    if tcol:work['_t']=pd.to_datetime(work[tcol],utc=True,errors='coerce')
    for _,z in work.groupby('region_group_id',sort=True):
        if '_t' in z:z=z.sort_values('_t')
        pos=z[z.label_m1plus_24h.eq(1)];neg=z[z.label_m1plus_24h.eq(0)]
        want_pos=min(len(pos),max(0,min(pos_cap,per_group)))
        def spread(x,n):
            if n<=0:return x.iloc[0:0]
            if len(x)<=n:return x
            ids=np.unique(np.round(np.linspace(0,len(x)-1,n)).astype(int))
            if len(ids)<n:
                extra=[i for i in range(len(x)) if i not in set(ids)][:n-len(ids)];ids=np.r_[ids,extra]
            return x.iloc[ids[:n]]
        p=spread(pos,want_pos);n=spread(neg,min(len(neg),per_group-len(p)));q=pd.concat([p,n])
        if len(q)<min(per_group,len(z)):
            rest=z[~z.index.isin(q.index)];q=pd.concat([q,spread(rest,min(per_group-len(q),len(rest)))])
        pieces.append(q)
    out=pd.concat(pieces,ignore_index=True) if pieces else work.iloc[0:0].copy()
    return out.drop(columns=['_t'],errors='ignore').sample(frac=1,random_state=seed).reset_index(drop=True)


def build_point_image_records(evidence_dir: str|Path, partition: str, per_group: int=0, pos_cap: int=4, seed: int=2026) -> pd.DataFrame:
    return temporal_group_cap(build_records(evidence_dir,partition),per_group,pos_cap,seed)


class FeatureSequenceDataset(Dataset):
    """Paper-style 40-step / 36-minute knowledge-informed state sequences."""
    def __init__(self,evidence_dir:str|Path,partition:str,n_features:int=10,per_group:int=0,pos_cap:int=4,seed:int=2026):
        d=Path(evidence_dir)/'data'/'derived'
        idx=pd.read_csv(d/'cdr_sequence_index.csv.gz',low_memory=False)
        idx=temporal_group_cap(idx[idx.partition.eq(partition)].copy(),per_group,pos_cap,seed)
        feat=pd.read_csv(d/'cdr_feature_metadata.csv.gz',low_memory=False)
        stats=json.loads((d/'cdr_feature_train_stats.json').read_text());names=FEATURES10 if int(n_features)==10 else FEATURES2
        self.names=names;self.seq_len=int(stats['seq_len']);self.step_minutes=int(stats.get('step_minutes',36));self.rows=idx.reset_index(drop=True)
        feat['time']=_parse_trec(feat.T_REC);feat['harpnum']=pd.to_numeric(feat.HARPNUM,errors='coerce').astype('Int64')
        feat=feat[feat.time.notna() & feat.harpnum.notna()].copy()
        for f in names:feat[f]=pd.to_numeric(feat[f],errors='coerce')
        st=stats['stats'];self.median=np.array([st[f]['median'] for f in names],np.float32);self.mean=np.array([st[f]['mean'] for f in names],np.float32);self.std=np.array([st[f]['std'] for f in names],np.float32)
        self.lookup={(int(r.harpnum),r.time):np.array([r[f] for f in names],np.float32) for _,r in feat.iterrows()}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i];h=int(r.harpnum)
        times=[pd.Timestamp(t) for t in str(r.sequence_times).split('|')]
        vals=[]
        for t in times:
            if t.tzinfo is None:t=t.tz_localize('UTC')
            x=self.lookup[(h,t)].copy();bad=~np.isfinite(x);x[bad]=self.median[bad];vals.append((x-self.mean)/self.std)
        return {'x':torch.from_numpy(np.stack(vals).astype(np.float32)),'y':torch.tensor(float(r.label_m1plus_24h),dtype=torch.float32),'group':str(r.region_group_id),'sample_id':str(r.sample_id)}


def feature_index(evidence_dir:str|Path,partition:str)->pd.DataFrame:
    d=Path(evidence_dir)/'data'/'derived';x=pd.read_csv(d/'cdr_sequence_index.csv.gz',low_memory=False)
    return x[x.partition.eq(partition)].copy().reset_index(drop=True)


def build_image_sequence_endpoints(evidence_dir:str|Path,partition:str,per_group:int=1,pos_cap:int=1,seed:int=2026):
    d=Path(evidence_dir)/'data'/'derived';idx=pd.read_csv(d/'cdr_sequence_index.csv.gz',low_memory=False)
    idx=idx[idx.partition.eq(partition)].copy();return temporal_group_cap(idx,per_group,pos_cap,seed)


class CachedImageSequenceDataset(Dataset):
    def __init__(self,cached_records:pd.DataFrame,endpoints:pd.DataFrame):
        self.end=endpoints.reset_index(drop=True);self.records=cached_records.reset_index(drop=True)
        self.pos={str(r.frame_id):i for i,r in self.records.iterrows()};self.base=MagnetogramDataset(self.records)
    def __len__(self):return len(self.end)
    def __getitem__(self,i):
        r=self.end.iloc[i];h=int(r.harpnum);times=[pd.Timestamp(t) for t in str(r.sequence_times).split('|')];frames=[]
        for t in times:
            if t.tzinfo is None:t=t.tz_localize('UTC')
            fid=f'H{h}_{t.strftime("%Y%m%dT%H%M%SZ")}';frames.append(self.base[self.pos[fid]]['x'])
        return {'x':torch.stack(frames,0),'y':torch.tensor(float(r.label_m1plus_24h),dtype=torch.float32),'group':str(r.region_group_id),'sample_id':str(r.sample_id)}


def cache_image_sequences(evidence_dir:str|Path,partition:str,cache_dir:str|Path,seq_len:int=40,per_group:int=1,pos_cap:int=1,seed:int=2026,workers:int=12):
    """Cache exact 40-frame / 36-minute SHARP LOS sequences.

    Geometry and URLs come from the native-12m CDR metadata artifact. Only the
    40 required frames for capped sequence endpoints are downloaded.
    """
    d=Path(evidence_dir)/'data'/'derived';end=build_image_sequence_endpoints(evidence_dir,partition,per_group,pos_cap,seed)
    if not len(end):raise RuntimeError(f'no image sequences for {partition}')
    if int(end.sequence_length.iloc[0])!=int(seq_len):raise ValueError(f'evidence sequence length is {end.sequence_length.iloc[0]}, requested {seq_len}')
    meta=pd.read_csv(d/'cdr_feature_metadata.csv.gz',low_memory=False);meta['time']=_parse_trec(meta.T_REC);meta['harpnum']=pd.to_numeric(meta.HARPNUM,errors='coerce').astype('Int64')
    meta=meta[meta.time.notna() & meta.harpnum.notna()].copy();lookup={(int(r.harpnum),r.time):r for _,r in meta.iterrows()}
    keys=set()
    for _,r in end.iterrows():
        h=int(r.harpnum)
        for ts in str(r.sequence_times).split('|'):
            t=pd.Timestamp(ts);t=t.tz_localize('UTC') if t.tzinfo is None else t;keys.add((h,t))
    rows=[]
    for h,t in sorted(keys,key=lambda z:(z[1],z[0])):
        r=lookup[(h,t)];fid=f'H{h}_{t.strftime("%Y%m%dT%H%M%SZ")}'
        rows.append({'frame_id':fid,'sample_id':fid,'magnetogram_url':r.segment_magnetogram,'CDELT1':r.CDELT1,'CDELT2':r.CDELT2,'RSUN_REF':r.RSUN_REF,
                     'label_m1plus_24h':0,'latitude_deg':float(r.LAT_FWT) if pd.notna(r.LAT_FWT) else 0.0,'region_group_id':'CDR_FRAME'})
    rec=pd.DataFrame(rows);rec=cache_records(rec,cache_dir,workers);return rec,end
