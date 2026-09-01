from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from preprocess import PreprocessConfig, preprocess_fits
from fit_cache import index_local_fits, is_fits_payload

COMMON = Path(__file__).resolve().parents[1] / 'common'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from jsoc_time import parse_jsoc_trec_to_utc


def _parse_jsoc_time(s: pd.Series) -> pd.Series:
    return parse_jsoc_trec_to_utc(s)


def build_records(evidence_dir: str | Path, partition: str) -> pd.DataFrame:
    d = Path(evidence_dir) / 'data' / 'derived'
    receipt_path = d / 'tai_repair_audit.json'
    if not receipt_path.exists():
        raise RuntimeError(
            'Evidence is missing tai_repair_audit.json; run the fail-closed '
            'historical TAI-to-UTC repair before model loading.'
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Invalid TAI repair receipt: {receipt_path}') from exc
    if str(receipt.get('status', '')) != 'PASS':
        raise RuntimeError(f'Evidence TAI repair did not pass: {receipt_path}')
    man = pd.read_csv(d / 'training_manifest.csv.gz', low_memory=False)
    meta = pd.read_csv(d / 'sharp_metadata.csv.gz', low_memory=False)
    man = man[man.partition.eq(partition) & man.label_m1plus_24h.notna()].copy()
    man['join_time'] = pd.to_datetime(man.t_rec, utc=True, errors='coerce').dt.floor('h')
    meta['join_time'] = _parse_jsoc_time(meta.T_REC).dt.floor('h')
    meta['harpnum'] = pd.to_numeric(meta.HARPNUM, errors='coerce')
    keep = ['harpnum', 'join_time', 'CDELT1', 'CDELT2', 'RSUN_REF']
    meta = meta[keep].dropna(subset=['harpnum', 'join_time']).drop_duplicates(['harpnum', 'join_time'])
    out = man.merge(meta, on=['harpnum', 'join_time'], how='left', validate='many_to_one')
    required = ['magnetogram_url', 'CDELT1', 'CDELT2', 'RSUN_REF']
    if out[required].isna().any().any():
        bad = out[out[required].isna().any(axis=1)][['sample_id'] + required]
        raise RuntimeError(f'Missing image/geometry metadata for {len(bad)} samples; first rows:\n{bad.head()}')
    return out.reset_index(drop=True)


def deterministic_smoke_subset(df: pd.DataFrame, n: int, seed: int = 17) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy().reset_index(drop=True)
    pos = df[df.label_m1plus_24h.eq(1)]
    neg = df[df.label_m1plus_24h.eq(0)]
    np = min(len(pos), max(1, n // 4)); nn = n - np
    p = pos.sample(n=np, random_state=seed) if np else pos
    q = neg.sample(n=min(nn, len(neg)), random_state=seed + 1)
    z = pd.concat([p, q], ignore_index=True)
    if len(z) < n:
        rest = df[~df.sample_id.isin(z.sample_id)].sample(n=min(n-len(z), len(df)-len(z)), random_state=seed+2)
        z = pd.concat([z, rest], ignore_index=True)
    return z.sample(frac=1, random_state=seed+3).reset_index(drop=True)


def _download_one(row: dict, cache_dir: Path, max_attempts: int = 5) -> tuple[str, str, int]:
    """Download one immutable JSOC FITS record with bounded retry/backoff."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sid = str(row['sample_id']); path = cache_dir / f'{sid}.fits'
    if path.exists() and path.stat().st_size > 2880:
        return sid, str(path), path.stat().st_size
    url = str(row['magnetogram_url']); last = None
    with requests.Session() as s:
        s.headers.update({'User-Agent': 'IRIS-ISEF-research/1.0'})
        for attempt in range(max_attempts):
            try:
                r = s.get(url, timeout=180, allow_redirects=True)
                if r.status_code in {429, 500, 502, 503, 504}:
                    retry_after = r.headers.get('Retry-After')
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 3.0 * (2 ** attempt))
                    last = RuntimeError(f'HTTP {r.status_code}'); time.sleep(delay); continue
                r.raise_for_status()
                if len(r.content) <= 2880 or not r.content.startswith(b'SIMPLE'):
                    raise RuntimeError(f'Not a valid FITS payload: {len(r.content)} bytes')
                tmp = path.with_suffix('.part'); tmp.write_bytes(r.content); tmp.replace(path)
                return sid, str(path), path.stat().st_size
            except Exception as e:
                last = e
                if attempt + 1 < max_attempts: time.sleep(min(60.0, 3.0 * (2 ** attempt)))
    raise RuntimeError(f'{sid}: failed to download {url}: {last}')


def cache_records(df: pd.DataFrame, cache_dir: str | Path, workers: int = 8) -> pd.DataFrame:
    """Cache all records; retry transient JSOC failures serially before aborting."""
    local_source = os.environ.get('IRIS_FITS_SOURCE', '').strip()
    if os.environ.get('IRIS_REQUIRE_LOCAL_FITS', '0') == '1' and not local_source:
        raise RuntimeError(
            'IRIS_REQUIRE_LOCAL_FITS=1 but IRIS_FITS_SOURCE is unset; '
            'materialize and verify the real FITS cache before training.'
        )
    if local_source:
        index = index_local_fits(local_source)
        missing = []
        invalid = []
        paths = {}
        for row in df.to_dict('records'):
            sid = str(row['sample_id'])
            path = index.get(sid)
            if path is None:
                missing.append(sid)
            elif not is_fits_payload(path):
                invalid.append(sid)
            else:
                paths[sid] = (str(path.resolve()), int(path.stat().st_size))
        if missing or invalid:
            raise RuntimeError(
                f'Local FITS cache is incomplete for this stage: '
                f'missing={len(missing)}, invalid={len(invalid)}; '
                f'examples={(missing + invalid)[:5]}'
            )
        out = df.copy()
        out['fits_path'] = out.sample_id.map(lambda x: paths[str(x)][0])
        out['fits_bytes'] = out.sample_id.map(lambda x: paths[str(x)][1])
        print(f'using verified local FITS cache: {len(out)} records from {local_source}', flush=True)
        return out
    cache = Path(cache_dir); rows = df.to_dict('records'); got = {}; failed = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as ex:
        futs = {ex.submit(_download_one, r, cache, 5): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                sid, path, size = fut.result(); got[sid] = (path, size)
            except Exception as e:
                failed.append((row, e))
            if i % 250 == 0 or i == len(rows):
                print(f'cached first-pass {i}/{len(rows)}; transient_failures={len(failed)}', flush=True)
    if failed:
        print(f'retrying {len(failed)} JSOC records serially with extended backoff', flush=True); still=[]
        for i,(row,first_error) in enumerate(failed,1):
            try:
                sid,path,size=_download_one(row,cache,12); got[sid]=(path,size)
            except Exception as e:
                still.append((str(row['sample_id']),str(first_error),str(e)))
            if i%25==0 or i==len(failed): print(f'serial retry {i}/{len(failed)}; unresolved={len(still)}',flush=True)
        if still: raise RuntimeError(f'{len(still)} JSOC records remain unavailable after extended retries; first failures={still[:5]}')
    out=df.copy(); out['fits_path']=out.sample_id.map(lambda x:got[str(x)][0]); out['fits_bytes']=out.sample_id.map(lambda x:got[str(x)][1]); return out


@dataclass
class DatasetConfig:
    preprocess: PreprocessConfig = PreprocessConfig()


class MagnetogramDataset(Dataset):
    def __init__(self, records: pd.DataFrame, cfg: DatasetConfig = DatasetConfig()):
        self.records = records.reset_index(drop=True); self.cfg = cfg
    def __len__(self): return len(self.records)
    def __getitem__(self, i: int):
        r=self.records.iloc[i]
        x,raw=preprocess_fits(r.fits_path,float(r.CDELT1),float(r.CDELT2),float(r.RSUN_REF),self.cfg.preprocess)
        return {'x':x.float(),'raw_gauss':raw.float(),'y':torch.tensor(float(r.label_m1plus_24h),dtype=torch.float32),'latitude':torch.tensor(float(r.latitude_deg),dtype=torch.float32),'group':str(r.region_group_id),'sample_id':str(r.sample_id)}
