#!/usr/bin/env python3
from __future__ import annotations
import json, time
from pathlib import Path
from datetime import datetime, timedelta
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


def jsoc_chunk(a:datetime,b:datetime):
    # SHARP prime keys are HARPNUM,T_REC. The 1-hour issue cadence is frozen before
    # any forecasting result exists; it reduces strongly overlapping 12-min samples
    # while retaining six issue times across a 6-hour interval and 24/day.
    dur_hours=max(1,int((b-a).total_seconds()/3600))
    ds=f"hmi.sharp_cea_720s[][{a.strftime('%Y.%m.%d_%H:%M:%S')}_TAI/{dur_hours}h@{ISSUE_CADENCE}]"
    params={'op':'rs_list','ds':ds,'key':','.join(KEYS),'seg':SEG}
    failures=[]
    for base in BASES:
        for attempt in range(3):
            try:
                r=requests.get(base,params=params,timeout=180,allow_redirects=True)
                r.raise_for_status(); j=r.json()
                if int(j.get('status',0))!=0:
                    raise RuntimeError(f"JSOC status={j.get('status')} error={j.get('error')}")
                return ds,j,base
            except Exception as e:
                failures.append(f'{base} attempt {attempt+1}: {e}')
                time.sleep(3*(attempt+1))
    raise RuntimeError(f"JSOC query failed for {a}..{b}: {' | '.join(failures)}")


def rows_from(j):
    count=int(j.get('count',0) or 0)
    if count==0: return []
    kw={x['name']:x.get('values',[]) for x in j.get('keywords',[])}
    seg={x['name']:x.get('values',[]) for x in j.get('segments',[])}
    rows=[]
    for i in range(count):
        row={k:(kw.get(k,[None]*count)[i] if i<len(kw.get(k,[])) else None) for k in KEYS}
        sval=seg.get(SEG,[None]*count)
        path=sval[i] if i<len(sval) else None
        if path and str(path).startswith('/'):
            path='http://jsoc.stanford.edu'+str(path)
        row['segment_magnetogram']=path
        rows.append(row)
    return rows


def main():
    allrows=[]; queries=[]
    a=START
    # 31-day chunks at the frozen hourly issue cadence keep responses compact.
    while a<END:
        b=min(a+timedelta(days=31),END)
        print('query',a,b,flush=True)
        ds,j,base=jsoc_chunk(a,b)
        rs=rows_from(j); allrows.extend(rs)
        queries.append({'recordset':ds,'count':len(rs),'endpoint':base,'issue_cadence':ISSUE_CADENCE})
        a=b
    df=pd.DataFrame(allrows)
    if df.empty: raise SystemExit('No SHARP metadata returned')
    df=df.drop_duplicates(subset=['HARPNUM','T_REC']).sort_values(['T_REC','HARPNUM'])
    out=DER/'sharp_metadata.csv.gz'; df.to_csv(out,index=False,compression='gzip')
    (DER/'sharp_query_log.json').write_text(json.dumps({'queries':queries,'rows_after_dedup':len(df),'columns':list(df.columns),'issue_cadence':ISSUE_CADENCE},indent=2)+'\n')
    print(f'SHARP rows: {len(df):,}; unique HARPs: {df.HARPNUM.nunique():,}; cadence={ISSUE_CADENCE}')
    print(out)

if __name__=='__main__': main()
