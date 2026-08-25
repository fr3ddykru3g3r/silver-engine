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
BASE='https://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info'
KEYS=['T_REC','HARPNUM','NOAA_AR','NOAA_NUM','NOAA_ARS','LON_FWT','LAT_FWT','QUALITY','USFLUX','R_VALUE','CDELT1','CDELT2','RSUN_REF','T_FIRST','T_LAST']
SEG='magnetogram'


def jsoc_chunk(a:datetime,b:datetime):
    # SHARP prime keys are HARPNUM,T_REC. Empty first selector means all HARPs.
    ds=f"hmi.sharp_cea_720s[][{a.strftime('%Y.%m.%d_%H:%M:%S')}_TAI-{b.strftime('%Y.%m.%d_%H:%M:%S')}_TAI]"
    params={'op':'rs_list','ds':ds,'key':','.join(KEYS),'seg':SEG}
    last=None
    for attempt in range(5):
        try:
            r=requests.get(BASE,params=params,timeout=180)
            r.raise_for_status(); j=r.json()
            if int(j.get('status',0))!=0:
                raise RuntimeError(f"JSOC status={j.get('status')} error={j.get('error')}")
            return ds,j
        except Exception as e:
            last=e; time.sleep(3*(attempt+1))
    raise RuntimeError(f'JSOC query failed for {a}..{b}: {last}')


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
            path='https://jsoc.stanford.edu'+str(path)
        row['segment_magnetogram']=path
        rows.append(row)
    return rows


def main():
    allrows=[]; queries=[]
    a=START
    # Seven-day chunks keep jsoc_info responses bounded while preserving native 12-min cadence.
    while a<END:
        b=min(a+timedelta(days=7),END)
        print('query',a,b,flush=True)
        ds,j=jsoc_chunk(a,b)
        rs=rows_from(j); allrows.extend(rs)
        queries.append({'recordset':ds,'count':len(rs)})
        a=b
    df=pd.DataFrame(allrows)
    if df.empty: raise SystemExit('No SHARP metadata returned')
    # Deduplicate inclusive chunk-boundary timestamps if JSOC returns them twice.
    df=df.drop_duplicates(subset=['HARPNUM','T_REC']).sort_values(['T_REC','HARPNUM'])
    out=DER/'sharp_metadata.csv.gz'; df.to_csv(out,index=False,compression='gzip')
    (DER/'sharp_query_log.json').write_text(json.dumps({'queries':queries,'rows_after_dedup':len(df),'columns':list(df.columns)},indent=2)+'\n')
    print(f'SHARP rows: {len(df):,}; unique HARPs: {df.HARPNUM.nunique():,}')
    print(out)

if __name__=='__main__': main()
