#!/usr/bin/env python3
from __future__ import annotations
import json, time, subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'; DER.mkdir(parents=True,exist_ok=True)
START=datetime.fromisoformat('2025-08-25')
END=datetime.fromisoformat('2026-08-24')  # exclusive
BASES=['http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info','https://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info']
KEYS=['T_REC','HARPNUM','NOAA_AR','NOAA_NUM','NOAA_ARS','LON_FWT','LAT_FWT','QUALITY','USFLUX','R_VALUE','CDELT1','CDELT2','RSUN_REF','T_FIRST','T_LAST']
SEG='magnetogram'; ISSUE_CADENCE='1h'


def make_query(harp:int):
    dur_hours=max(1,int((END-START).total_seconds()/3600))
    ds=f"hmi.sharp_cea_720s[{harp}][{START.strftime('%Y.%m.%d_%H:%M:%S')}_TAI/{dur_hours}h@{ISSUE_CADENCE}]"
    return ds,{'op':'rs_list','ds':ds,'key':','.join(KEYS),'seg':SEG}


def valid_json(j):
    if int(j.get('status',0))!=0: raise RuntimeError(f"JSOC status={j.get('status')} error={j.get('error')}")
    return j


def query_harp(harp:int):
    ds,params=make_query(harp); failures=[]
    for base in BASES:
        for attempt in range(2):
            try:
                r=requests.get(base,params=params,timeout=90,allow_redirects=True)
                r.raise_for_status(); return harp,ds,valid_json(r.json()),base,None
            except Exception as e:
                failures.append(f'{base} attempt {attempt+1}: {e}'); time.sleep(1.5*(attempt+1))
    return harp,ds,None,None,' | '.join(failures)


def curl_recover(harp:int):
    ds,params=make_query(harp)
    url=BASES[0]+'?'+urlencode(params)
    last=''
    for attempt in range(5):
        try:
            cp=subprocess.run(['curl','-L','--retry','3','--retry-delay','2','--max-time','120','-fsS',url],capture_output=True,text=True,timeout=140)
            if cp.returncode!=0: raise RuntimeError(cp.stderr.strip() or f'curl rc={cp.returncode}')
            return harp,ds,valid_json(json.loads(cp.stdout)),BASES[0]+' (curl recovery)',None
        except Exception as e:
            last=str(e); time.sleep(3*(attempt+1))
    return harp,ds,None,None,last


def rows_from(j):
    count=int(j.get('count',0) or 0)
    if count==0:return []
    kw={x['name']:x.get('values',[]) for x in j.get('keywords',[])}; seg={x['name']:x.get('values',[]) for x in j.get('segments',[])}
    rows=[]
    for i in range(count):
        row={k:(kw.get(k,[None]*count)[i] if i<len(kw.get(k,[])) else None) for k in KEYS}
        sval=seg.get(SEG,[None]*count); path=sval[i] if i<len(sval) else None
        if path and str(path).startswith('/'):path='http://jsoc.stanford.edu'+str(path)
        row['segment_magnetogram']=path; rows.append(row)
    return rows


def main():
    active_path=DER/'active_harps.csv'
    if not active_path.exists():raise SystemExit('active_harps.csv missing; run fetch_gate0.py first')
    active=pd.read_csv(active_path); harps=sorted(set(pd.to_numeric(active.harpnum,errors='coerce').dropna().astype(int)))
    print(f'Querying {len(harps)} SRS-defined NOAA-associated HARPs at {ISSUE_CADENCE} cadence',flush=True)
    results={}; failed=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(query_harp,h):h for h in harps}
        for n,fut in enumerate(as_completed(futs),1):
            h,ds,j,base,err=fut.result()
            if err:failed.append(h)
            else:results[h]=(ds,j,base)
            if n%25==0 or n==len(harps):print(f'first pass {n}/{len(harps)}; recovered={len(results)}; pending={len(failed)}',flush=True)
    if failed:
        print(f'Curl-recovering {len(failed)} transient failures serially: {failed}',flush=True)
        still=[]
        for h in failed:
            hh,ds,j,base,err=curl_recover(h)
            if err:
                print(f'UNRECOVERED HARP {h}: {err}',flush=True); still.append({'harpnum':h,'error':err})
            else:
                results[h]=(ds,j,base); print(f'recovered HARP {h}',flush=True)
        failed=still
    pd.DataFrame(failed).to_csv(DER/'sharp_query_failures.csv',index=False)
    if failed:raise SystemExit(f'{len(failed)} HARP queries failed after curl recovery; refusing incomplete manifest')

    allrows=[]; queries=[]
    for h in harps:
        ds,j,base=results[h]; rs=rows_from(j); allrows.extend(rs)
        queries.append({'harpnum':h,'recordset':ds,'count':len(rs),'endpoint':base,'issue_cadence':ISSUE_CADENCE})
    df=pd.DataFrame(allrows)
    if df.empty:raise SystemExit('No SHARP metadata returned')
    df=df.drop_duplicates(subset=['HARPNUM','T_REC']).sort_values(['T_REC','HARPNUM'])
    out=DER/'sharp_metadata.csv.gz'; df.to_csv(out,index=False,compression='gzip')
    (DER/'sharp_query_log.json').write_text(json.dumps({'queries':queries,'rows_after_dedup':len(df),'columns':list(df.columns),'issue_cadence':ISSUE_CADENCE,'active_harps_expected':len(harps),'active_harps_returned':int(df.HARPNUM.nunique())},indent=2)+'\n')
    print(f'SHARP rows: {len(df):,}; unique HARPs: {df.HARPNUM.nunique():,}; cadence={ISSUE_CADENCE}',flush=True)

if __name__=='__main__':main()
