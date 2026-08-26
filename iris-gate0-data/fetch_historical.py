#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'; DER=ROOT/'data'/'derived'; RAW.mkdir(parents=True,exist_ok=True); DER.mkdir(parents=True,exist_ok=True)
START='2010-05-01'; END='2026-08-23'
NCEI='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/sci_xrsf-l2-flrpt_geo_s19950103_e20260823_v1-0-1.csv'
HARP='http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt'

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def get(url,dest):
    last=None
    for i in range(5):
        try:
            r=requests.get(url,timeout=180,allow_redirects=True); r.raise_for_status(); Path(dest).write_bytes(r.content)
            return {'url':url,'path':str(Path(dest).relative_to(ROOT)),'bytes':Path(dest).stat().st_size,'sha256':sha256(dest)}
        except Exception as e:last=e; time.sleep(2*(i+1))
    raise last

def parse_map(path):
    rows=[]
    for line in Path(path).read_text(errors='ignore').splitlines():
        s=line.strip()
        if not s or s.startswith('#'):continue
        nums=[int(x) for x in re.findall(r'\d+',s)]
        if not nums:continue
        h=nums[0]; ns=sorted(set(n for n in nums[1:] if n>=10000))
        rows.append({'harpnum':h,'noaa_ars':';'.join(map(str,ns)),'source_line':s})
    return pd.DataFrame(rows).drop_duplicates('harpnum')

def main():
    ledger=[]; fr=RAW/'goes_mission.csv'; hm=RAW/'all_harps_with_noaa_ars.txt'
    ledger.append(get(NCEI,fr))
    try:ledger.append(get(HARP,hm))
    except Exception:ledger.append(get(HARP.replace('http://','https://'),hm))
    f=pd.read_csv(fr,low_memory=False); f['start_time']=pd.to_datetime(f['start_time'],utc=True,errors='coerce')
    mag=f['flare_class'].astype(str).str.extract(r'([MX])(\d+(?:\.\d+)?)',expand=True)
    mx=mag[0].isin(['M','X']) & pd.to_numeric(mag[1],errors='coerce').ge(1.0)
    sub=f[mx & f.start_time.ge(pd.Timestamp(START,tz='UTC')) & f.start_time.lt(pd.Timestamp(END,tz='UTC')+pd.Timedelta(days=1))].copy()
    sub.to_csv(DER/'goes_m1plus_interval.csv',index=False)
    m=parse_map(hm); m.to_csv(DER/'harp_noaa_mapping.csv',index=False)
    long=[]
    for _,r in m.iterrows():
        for a in str(r.noaa_ars).split(';'):
            if a:
                n=int(a); long.append({'harpnum':int(r.harpnum),'noaa_ar':n,'noaa_last4':n%10000})
    pd.DataFrame(long).to_csv(DER/'harp_noaa_long.csv',index=False)
    ars=pd.to_numeric(sub.active_region,errors='coerce').dropna().astype(int)
    summary={'interval_start':START,'interval_end':END,'m1plus_events':len(sub),'m1plus_events_with_ar':len(ars),'unique_reported_active_regions':int(ars.nunique()),'mapped_harps':int(m.harpnum.nunique())}
    (DER/'collection_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (DER/'source_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
