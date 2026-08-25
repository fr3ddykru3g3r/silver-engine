#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, re, time
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
DER = ROOT / 'data' / 'derived'
RAW.mkdir(parents=True, exist_ok=True)
DER.mkdir(parents=True, exist_ok=True)

START = date.fromisoformat('2025-08-25')
END = date.fromisoformat('2026-08-23')
NCEI_DIR = 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/'
NCEI_MISSION = NCEI_DIR + 'sci_xrsf-l2-flrpt_geo_s19950103_e20260823_v1-0-1.csv'
HARP_MAP = 'http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt'
SRS_ROOT = 'https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries'


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def get(url: str, dest: Path, timeout=120):
    last=None
    for i in range(4):
        try:
            r=requests.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status(); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(r.content)
            return {'url':url,'path':str(dest.relative_to(ROOT)),'sha256':sha256(dest),'bytes':dest.stat().st_size}
        except Exception as e:
            last=e; time.sleep(2*(i+1))
    raise last


def parse_harp_map(path: Path):
    rows=[]
    for line in path.read_text(errors='ignore').splitlines():
        s=line.strip()
        if not s or s.startswith('#'): continue
        nums=[int(x) for x in re.findall(r'\d+',s)]
        if not nums: continue
        harp=nums[0]
        noaas=[n for n in nums[1:] if n>=10000]
        rows.append({'harpnum':harp,'noaa_ars':';'.join(map(str,sorted(set(noaas)))),'source_line':s})
    return pd.DataFrame(rows).drop_duplicates('harpnum')


def parse_srs_regions(text: str):
    # Region rows start with a 4- or 5-digit NOAA number and a heliographic position.
    # This captures both sunspot and plage rows but not report/SRS sequence numbers.
    out=[]
    for ln in text.splitlines():
        m=re.match(r'^\s*(\d{4,5})\s+([NS]\d{2}[EW]\d{2})\b',ln)
        if m: out.append((int(m.group(1)),m.group(2)))
    return out


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
    cols={c.lower():c for c in f.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        for c in f.columns:
            lc=c.lower()
            if any(n in lc for n in names): return c
        raise KeyError(names)
    cstart=pick('start_time'); cclass=pick('flare_class','class'); car=pick('active_region','active region')
    f[cstart]=pd.to_datetime(f[cstart], utc=True, errors='coerce')
    mag=f[cclass].astype(str).str.extract(r'([MX])(\d+(?:\.\d+)?)', expand=True)
    is_mx=mag[0].isin(['M','X']) & mag[1].astype(float).ge(1.0)
    s=pd.Timestamp(START,tz='UTC'); e=pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1)
    sub=f[is_mx & f[cstart].ge(s) & f[cstart].lt(e)].copy()
    sub.to_csv(DER/'goes_m1plus_interval.csv',index=False)

    hm=parse_harp_map(harp_raw)
    hm.to_csv(DER/'harp_noaa_mapping.csv',index=False)
    long=[]
    for _,r in hm.iterrows():
        for a in str(r.noaa_ars).split(';'):
            if a:
                n=int(a); long.append({'harpnum':int(r.harpnum),'noaa_ar':n,'noaa_last4':n%10000})
    longdf=pd.DataFrame(long)
    longdf.to_csv(DER/'harp_noaa_long.csv',index=False)

    # Complete daily NOAA Solar Region Summary census for the frozen interval.
    srs_rows=[]; missing=[]; d=START
    while d<=END:
        url=f'{SRS_ROOT}/{d:%Y}/{d:%m}/{d:%Y%m%d}SRS.txt'
        p=RAW/'srs'/f'{d:%Y%m%d}SRS.txt'
        try:
            rec=get(url,p,timeout=40); ledger.append(rec)
            for reported,loc in parse_srs_regions(p.read_text(errors='ignore')):
                # Modern SRS commonly prints last four digits (e.g. 4366 for NOAA 14366).
                cands=longdf[longdf.noaa_last4.eq(reported)] if reported<10000 else longdf[longdf.noaa_ar.eq(reported)]
                canon=sorted(cands.noaa_ar.unique().tolist())
                srs_rows.append({'report_date':d.isoformat(),'reported_region':reported,'location':loc,
                                 'canonical_noaa_candidates':';'.join(map(str,canon)),
                                 'resolution':'UNIQUE' if len(canon)==1 else ('AMBIGUOUS' if len(canon)>1 else 'UNMAPPED')})
        except Exception as ex:
            missing.append({'date':d.isoformat(),'url':url,'error':str(ex)})
        d += timedelta(days=1)
    srs=pd.DataFrame(srs_rows)
    srs.to_csv(DER/'srs_region_census.csv',index=False)
    pd.DataFrame(missing).to_csv(DER/'srs_missing_reports.csv',index=False)

    # Active HARP set = every HARP linked to a NOAA region seen in the frozen interval.
    unique_noaa=set()
    for x in srs.canonical_noaa_candidates.dropna().astype(str):
        if x and ';' not in x: unique_noaa.add(int(x))
    active=longdf[longdf.noaa_ar.isin(unique_noaa)].copy().sort_values(['harpnum','noaa_ar'])
    active.to_csv(DER/'active_harps.csv',index=False)

    ars=pd.to_numeric(sub[car], errors='coerce').dropna().astype(int)
    stats={'interval_start':START.isoformat(),'interval_end':END.isoformat(),'m1plus_events':int(len(sub)),
           'm1plus_events_with_ar':int(len(ars)),'unique_reported_active_regions':int(ars.nunique()),
           'srs_reports_expected':int((END-START).days+1),'srs_reports_missing':len(missing),
           'srs_unique_canonical_noaa_regions':len(unique_noaa),'active_harps_from_srs':int(active.harpnum.nunique()),
           'flare_columns':list(f.columns),'flare_start_col':cstart,'flare_class_col':cclass,'flare_ar_col':car}
    (DER/'collection_summary.json').write_text(json.dumps(stats,indent=2,default=str)+'\n')
    (DER/'source_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(stats,indent=2,default=str))
    if missing: print(f'WARNING: {len(missing)} SRS reports missing; see srs_missing_reports.csv')

if __name__=='__main__': main()
