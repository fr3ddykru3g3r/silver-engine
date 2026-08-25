#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'; DER.mkdir(parents=True,exist_ok=True)
START=datetime.fromisoformat('2025-08-25')
END=datetime.fromisoformat('2026-08-24')  # exclusive
BASES=['http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info','https://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info']
KEYS=['T_REC','HARPNUM','NOAA_AR','NOAA_NUM','NOAA_ARS','LON_FWT','LAT_FWT','QUALITY','USFLUX','R_VALUE','CDELT1','CDELT2','RSUN_REF','T_FIRST','T_LAST']
SEG='magnetogram'
ISSUE_CADENCE='1h'


def query_harp(harp:int):
    dur_hours=max(1,int((END-START).total_seconds()/3600))
    ds=f"hmi.sharp_cea_720s[{harp}][{START.strftime('%Y.%m.%d_%H:%M:%S')}_TAI/{dur_hours}h@{ISSUE_CADENCE}]"
    params={'op':'rs_list','ds':ds,'key':','.join(KEYS),'seg':SEG}
    failures=[]
    for base in BASES:
        for attempt in range(2):
            try:
                r=requests.get(base,params=params,timeout=90,allow_redirects=True)
                r.raise_for_status(); j=r.json()
                if int(j.get('status',0))!=0:
                    raise RuntimeError(f"JSOC status={j.get('status')} error={j.get('error')}")
                return harp,ds,j,base,None
            except Exception as e:
                failures.append(f'{base} attempt {attempt+1}: {e}'); time.sleep(2*(attempt+1))
    return harp,ds,None,None,' | '.join(failures)


def rows_from(j):
    count=int(j.get('count',0) or 0)
    if count==0: return []
    kw={x['name']:x.get('values',[]) for x in j.get('keywords',[])}
    seg={x['name']:x.get('values',[]) for x in j.get('segments',[])}
    rows=[]
    for i in range(count):
        row={k:(kw.get(k,[None]*count)[i] if i<len(kw.get(k,[])) else None) for k in KEYS}
        sval=seg.get(SEG,[None]*count); path=sval[i] if i<len(sval) else None
        if path and str(path).startswith('/'): path='http://jsoc.stanford.edu'+str(path)
        row['segment_magnetogram']=path
        rows.append(row)
    return rows


def main():
    active_path=DER/'active_harps.csv'
    if not active_path.exists(): raise SystemExit('active_harps.csv missing; run fetch_gate0.py first')
    active=pd.read_csv(active_path)
    harps=sorted(set(pd.to_numeric(active.harpnum,errors='coerce').dropna().astype(int)))
    print(f'Querying {len(harps)} SRS-defined NOAA-associated HARPs at {ISSUE_CADENCE} cadence',flush=True)
    allrows=[]; queries=[]; failed=[]
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs={ex.submit(query_harp,h):h for h in harps}
        for n,fut in enumerate(as_completed(futs),1):
            h,ds,j,base,err=fut.result()
            if err:
                failed.append({'harpnum':h,'error':err}); print(f'FAILED HARP {h}',flush=True); continue
            rs=rows_from(j); allrows.extend(rs)
            queries.append({'harpnum':h,'recordset':ds,'count':len(rs),'endpoint':base,'issue_cadence':ISSUE_CADENCE})
            if n%20==0 or n==len(harps): print(f'completed {n}/{len(harps)}; rows={len(allrows):,}',flush=True)
    pd.DataFrame(failed).to_csv(DER/'sharp_query_failures.csv',index=False)
    if failed:
        raise SystemExit(f'{len(failed)} HARP queries failed; refusing incomplete manifest')
    df=pd.DataFrame(allrows)
    if df.empty: raise SystemExit('No SHARP metadata returned')
    df=df.drop_duplicates(subset=['HARPNUM','T_REC']).sort_values(['T_REC','HARPNUM'])
    out=DER/'sharp_metadata.csv.gz'; df.to_csv(out,index=False,compression='gzip')
    (DER/'sharp_query_log.json').write_text(json.dumps({'queries':queries,'rows_after_dedup':len(df),'columns':list(df.columns),'issue_cadence':ISSUE_CADENCE,'active_harps_expected':len(harps),'active_harps_returned':int(df.HARPNUM.nunique())},indent=2)+'\n')
    print(f'SHARP rows: {len(df):,}; unique HARPs: {df.HARPNUM.nunique():,}; cadence={ISSUE_CADENCE}',flush=True)

if __name__=='__main__': main()
