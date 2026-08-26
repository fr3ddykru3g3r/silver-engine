from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from preprocess import PreprocessConfig, preprocess_fits


def _parse_jsoc_time(s: pd.Series) -> pd.Series:
    x=s.astype(str).str.replace('_TAI','',regex=False)
    x=x.str.replace(r'^(\d{4})\.(\d{2})\.(\d{2})_',r'\1-\2-\3T',regex=True)
    return pd.to_datetime(x,utc=True,errors='coerce')


def build_records(evidence_dir: str|Path,partition:str)->pd.DataFrame:
    d=Path(evidence_dir)/'data'/'derived';man=pd.read_csv(d/'training_manifest.csv.gz',low_memory=False);meta=pd.read_csv(d/'sharp_metadata.csv.gz',low_memory=False)
    man=man[man.partition.eq(partition)&man.label_m1plus_24h.notna()].copy();man['join_time']=pd.to_datetime(man.t_rec,utc=True,errors='coerce').dt.floor('h')
    meta['join_time']=_parse_jsoc_time(meta.T_REC).dt.floor('h');meta['harpnum']=pd.to_numeric(meta.HARPNUM,errors='coerce')
    keep=['harpnum','join_time','CDELT1','CDELT2','RSUN_REF'];meta=meta[keep].dropna(subset=['harpnum','join_time']).drop_duplicates(['harpnum','join_time'])
    out=man.merge(meta,on=['harpnum','join_time'],how='left',validate='many_to_one');required=['magnetogram_url','CDELT1','CDELT2','RSUN_REF']
    if out[required].isna().any().any():
        bad=out[out[required].isna().any(axis=1)][['sample_id']+required];raise RuntimeError(f'Missing image/geometry metadata for {len(bad)} samples; first rows:\n{bad.head()}')
    return out.reset_index(drop=True)


def deterministic_smoke_subset(df:pd.DataFrame,n:int,seed:int=17)->pd.DataFrame:
    if n<=0 or len(df)<=n:return df.copy().reset_index(drop=True)
    pos=df[df.label_m1plus_24h.eq(1)];neg=df[df.label_m1plus_24h.eq(0)];np=min(len(pos),max(1,n//4));nn=n-np
    p=pos.sample(n=np,random_state=seed) if np else pos;q=neg.sample(n=min(nn,len(neg)),random_state=seed+1);z=pd.concat([p,q],ignore_index=True)
    if len(z)<n:
        rest=df[~df.sample_id.isin(z.sample_id)].sample(n=min(n-len(z),len(df)-len(z)),random_state=seed+2);z=pd.concat([z,rest],ignore_index=True)
    return z.sample(frac=1,random_state=seed+3).reset_index(drop=True)


def _download_one(row:dict,cache_dir:Path,max_attempts:int=5)->tuple[str,str,int]:
    """Download an immutable JSOC FITS record with bounded retry/backoff."""
    cache_dir.mkdir(parents=True,exist_ok=True);sid=str(row['sample_id']);path=cache_dir/f'{sid}.fits'
    if path.exists() and path.stat().st_size>2880:return sid,str(path),path.stat().st_size
    url=str(row['magnetogram_url']);last=None
    with requests.Session() as s:
        s.headers.update({'User-Agent':'IRIS-ISEF-research/1.0'})
        for attempt in range(max_attempts):
            try:
                r=s.get(url,timeout=180,allow_redirects=True)
                if r.status_code in {429,500,502,503,504}:
                    ra=r.headers.get('Retry-After');delay=float(ra) if ra and ra.isdigit() else min(60.,3.*(2**attempt));last=RuntimeError(f'HTTP {r.status_code}');time.sleep(delay);continue
                r.raise_for_status()
                if len(r.content)<=2880 or not r.content.startswith(b'SIMPLE'):raise RuntimeError(f'Not a valid FITS payload: {len(r.content)} bytes')
                tmp=path.with_suffix('.part');tmp.write_bytes(r.content);tmp.replace(path);return sid,str(path),path.stat().st_size
            except Exception as e:
                last=e
                if attempt+1<max_attempts:time.sleep(min(60.,3.*(2**attempt)))
    raise RuntimeError(f'{sid}: failed to download {url}: {last}')


def cache_records(df:pd.DataFrame,cache_dir:str|Path,workers:int=8)->pd.DataFrame:
    """Cache all rows; retry only transient failures serially before aborting."""
    cache=Path(cache_dir);rows=df.to_dict('records');got={};failed=[]
    with ThreadPoolExecutor(max_workers=max(1,min(int(workers),8))) as ex:
        futs={ex.submit(_download_one,r,cache,5):r for r in rows}
        for i,fut in enumerate(as_completed(futs),1):
            row=futs[fut]
            try:sid,path,size=fut.result();got[sid]=(path,size)
            except Exception as e:failed.append((row,e))
            if i%250==0 or i==len(rows):print(f'cached first-pass {i}/{len(rows)}; transient_failures={len(failed)}',flush=True)
    if failed:
        print(f'retrying {len(failed)} JSOC records serially with extended backoff',flush=True);still=[]
        for i,(row,first_error) in enumerate(failed,1):
            try:sid,path,size=_download_one(row,cache,12);got[sid]=(path,size)
            except Exception as e:still.append((str(row['sample_id']),str(first_error),str(e)))
            if i%25==0 or i==len(failed):print(f'serial retry {i}/{len(failed)}; unresolved={len(still)}',flush=True)
        if still:raise RuntimeError(f'{len(still)} JSOC records remain unavailable after extended retries; first failures={still[:5]}')
    out=df.copy();out['fits_path']=out.sample_id.map(lambda x:got[str(x)][0]);out['fits_bytes']=out.sample_id.map(lambda x:got[str(x)][1]);return out


@dataclass
class DatasetConfig:
    preprocess:PreprocessConfig=PreprocessConfig()


class MagnetogramDataset(Dataset):
    def __init__(self,records:pd.DataFrame,cfg:DatasetConfig=DatasetConfig()):self.records=records.reset_index(drop=True);self.cfg=cfg
    def __len__(self):return len(self.records)
    def __getitem__(self,i:int):
        r=self.records.iloc[i];x,raw=preprocess_fits(r.fits_path,float(r.CDELT1),float(r.CDELT2),float(r.RSUN_REF),self.cfg.preprocess)
        return {'x':x.float(),'raw_gauss':raw.float(),'y':torch.tensor(float(r.label_m1plus_24h),dtype=torch.float32),'latitude':torch.tensor(float(r.latitude_deg),dtype=torch.float32),'group':str(r.region_group_id),'sample_id':str(r.sample_id)}
