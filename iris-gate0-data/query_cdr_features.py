#!/usr/bin/env python3
from __future__ import annotations

import json, time, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DER = ROOT / 'data' / 'derived'
BASES = ['http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info',
         'https://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info']
START = '2010.05.01_00:00:00_TAI'
DURATION_H = int((pd.Timestamp('2026-08-24') - pd.Timestamp('2010-05-01')).total_seconds() / 3600)
# Wu et al. collect samples every 36 minutes and use 40 time steps. Querying the
# native 12-minute SHARP grid lets every frozen hourly forecast issue time serve
# as the exact endpoint of a 40 x 36-minute history (36 min = 3 native frames).
CADENCE = '12m'
FEATURES = ['R_VALUE','AREA_ACR','TOTUSJZ','TOTUSJH','TOTPOT','ABSNJZH',
            'SAVNCPP','USFLUX','MEANPOT','SHRGT45']
KEYS = ['T_REC','HARPNUM','QUALITY','LON_FWT','LAT_FWT','CDELT1','CDELT2','RSUN_REF'] + FEATURES
SEG='magnetogram'


def make_query(harp: int):
    ds = f'hmi.sharp_cea_720s[{harp}][{START}/{DURATION_H}h@{CADENCE}]'
    return ds, {'op':'rs_list','ds':ds,'key':','.join(KEYS),'seg':SEG}


def valid_json(j):
    if int(j.get('status', 0)) != 0:
        raise RuntimeError(f"JSOC status={j.get('status')} error={j.get('error')}")
    return j


def query_harp(harp: int):
    ds, params = make_query(harp); failures=[]
    for base in BASES:
        for attempt in range(2):
            try:
                r=requests.get(base, params=params, timeout=180, allow_redirects=True)
                r.raise_for_status(); return harp, ds, valid_json(r.json()), base, None
            except Exception as e:
                failures.append(f'{base} attempt {attempt+1}: {e}')
                time.sleep(1.5*(attempt+1))
    return harp, ds, None, None, ' | '.join(failures)


def curl_recover(harp: int):
    ds, params = make_query(harp); url=BASES[0]+'?'+urlencode(params); last=''
    for attempt in range(5):
        try:
            cp=subprocess.run(['curl','-L','--retry','3','--retry-delay','2','--max-time','240','-fsS',url],
                              capture_output=True,text=True,timeout=260)
            if cp.returncode != 0: raise RuntimeError(cp.stderr.strip() or f'curl rc={cp.returncode}')
            return harp, ds, valid_json(json.loads(cp.stdout)), BASES[0]+' (curl recovery)', None
        except Exception as e:
            last=str(e); time.sleep(3*(attempt+1))
    return harp, ds, None, None, last


def rows_from(j):
    count=int(j.get('count',0) or 0)
    if count==0:return []
    kw={x['name']:x.get('values',[]) for x in j.get('keywords',[])}
    sg={x['name']:x.get('values',[]) for x in j.get('segments',[])}
    out=[]
    for i in range(count):
        row={k:(kw.get(k,[None]*count)[i] if i<len(kw.get(k,[])) else None) for k in KEYS}
        vals=sg.get(SEG,[None]*count); path=vals[i] if i<len(vals) else None
        if path and str(path).startswith('/'):path='http://jsoc.stanford.edu'+str(path)
        row['segment_magnetogram']=path
        out.append(row)
    return out


def main():
    man=pd.read_csv(DER/'training_manifest.csv.gz',low_memory=False)
    harps=sorted(set(pd.to_numeric(man.harpnum,errors='coerce').dropna().astype(int)))
    print(f'Querying {len(harps)} HARPs for CDR 12m metadata / exact 36m histories',flush=True)
    results={}; failed=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(query_harp,h):h for h in harps}
        for n,f in enumerate(as_completed(futs),1):
            h,ds,j,base,err=f.result()
            if err: failed.append(h)
            else: results[h]=(ds,j,base)
            if n%100==0 or n==len(harps):
                print(f'first pass {n}/{len(harps)}; ok={len(results)}; pending={len(failed)}',flush=True)
    still=[]
    for h in failed:
        hh,ds,j,base,err=curl_recover(h)
        if err: still.append({'harpnum':h,'error':err})
        else: results[h]=(ds,j,base)
    pd.DataFrame(still).to_csv(DER/'cdr_feature_query_failures.csv',index=False)
    if still: raise SystemExit(f'{len(still)} feature queries failed; refusing incomplete CDR feature data')
    rows=[]; qlog=[]
    for h in harps:
        ds,j,base=results[h]; rr=rows_from(j); rows.extend(rr)
        qlog.append({'harpnum':h,'recordset':ds,'count':len(rr),'endpoint':base})
    df=pd.DataFrame(rows).drop_duplicates(['HARPNUM','T_REC']).sort_values(['T_REC','HARPNUM'])
    if df.empty: raise SystemExit('No CDR features returned')
    df.to_csv(DER/'cdr_feature_metadata.csv.gz',index=False,compression='gzip')
    coverage={}
    for f in FEATURES:
        z=pd.to_numeric(df[f],errors='coerce')
        coverage[f]={'finite':int(z.notna().sum()),'fraction':float(z.notna().mean())}
    report={'rows':len(df),'harps':int(pd.to_numeric(df.HARPNUM,errors='coerce').nunique()),
            'features':FEATURES,'coverage':coverage,'native_cadence':CADENCE,
            'cdr_sequence_cadence':'36m','cdr_sequence_steps':40,
            'source_series':'hmi.sharp_cea_720s',
            'magnetogram_urls':int(df.segment_magnetogram.notna().sum())}
    (DER/'cdr_feature_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (DER/'cdr_feature_query_log.json').write_text(json.dumps({'queries':qlog},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
