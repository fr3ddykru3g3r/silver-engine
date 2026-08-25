#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, re
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
DER = ROOT / 'data' / 'derived'
RAW.mkdir(parents=True, exist_ok=True)
DER.mkdir(parents=True, exist_ok=True)

START = '2025-08-25'
END = '2026-08-23'
NCEI_DIR = 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/'
NCEI_MISSION = NCEI_DIR + 'sci_xrsf-l2-flrpt_geo_s19950103_e20260823_v1-0-1.csv'
HARP_MAP = 'http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt'


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def get(url: str, dest: Path):
    r=requests.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status(); dest.write_bytes(r.content)
    return {'url':url,'path':str(dest.relative_to(ROOT)),'sha256':sha256(dest),'bytes':dest.stat().st_size}


def parse_harp_map(path: Path):
    rows=[]
    for line in path.read_text(errors='ignore').splitlines():
        s=line.strip()
        if not s or s.startswith('#'): continue
        nums=[int(x) for x in re.findall(r'\d+',s)]
        if not nums: continue
        harp=nums[0]
        noaas=[]
        for n in nums[1:]:
            if n>=10000: noaas.append(n)
        rows.append({'harpnum':harp,'noaa_ars':';'.join(map(str,sorted(set(noaas)))),'source_line':s})
    return pd.DataFrame(rows).drop_duplicates('harpnum')


def main():
    ledger=[]
    flare_raw=RAW/'goes_mission.csv'
    harp_raw=RAW/'all_harps_with_noaa_ars.txt'
    ledger.append(get(NCEI_MISSION, flare_raw))
    try:
        ledger.append(get(HARP_MAP, harp_raw))
    except Exception as e:
        print('HARP map HTTP fetch failed, retrying HTTPS:', e)
        ledger.append(get(HARP_MAP.replace('http://','https://'), harp_raw))

    f=pd.read_csv(flare_raw, low_memory=False)
    # NCEI schema names may evolve; normalize expected fields by substring search.
    cols={c.lower():c for c in f.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        for c in f.columns:
            lc=c.lower()
            if any(n in lc for n in names): return c
        raise KeyError(names)
    cstart=pick('start_time')
    cclass=pick('flare_class','class')
    car=pick('active_region','active region')
    f[cstart]=pd.to_datetime(f[cstart], utc=True, errors='coerce')
    mag=f[cclass].astype(str).str.extract(r'([MX])(\d+(?:\.\d+)?)', expand=True)
    is_mx=mag[0].isin(['M','X']) & mag[1].astype(float).ge(1.0)
    s=pd.Timestamp(START,tz='UTC'); e=pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1)
    sub=f[is_mx & f[cstart].ge(s) & f[cstart].lt(e)].copy()
    sub.to_csv(DER/'goes_m1plus_interval.csv',index=False)

    hm=parse_harp_map(harp_raw)
    hm.to_csv(DER/'harp_noaa_mapping.csv',index=False)
    # NOAA's event reports can sometimes abbreviate 5-digit ARs to the last four digits;
    # retain both exact and last-4 keys for audit, never silently relabel.
    long=[]
    for _,r in hm.iterrows():
        for a in str(r.noaa_ars).split(';'):
            if a:
                n=int(a); long.append({'harpnum':int(r.harpnum),'noaa_ar':n,'noaa_last4':n%10000})
    pd.DataFrame(long).to_csv(DER/'harp_noaa_long.csv',index=False)

    ars=pd.to_numeric(sub[car], errors='coerce').dropna().astype(int)
    stats={'interval_start':START,'interval_end':END,'m1plus_events':int(len(sub)),
           'm1plus_events_with_ar':int(ars.notna().sum()),'unique_reported_active_regions':int(ars.nunique()),
           'flare_columns':list(f.columns),'flare_start_col':cstart,'flare_class_col':cclass,'flare_ar_col':car}
    (DER/'collection_summary.json').write_text(json.dumps(stats,indent=2,default=str)+'\n')
    (DER/'source_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(stats,indent=2,default=str))

if __name__=='__main__': main()
