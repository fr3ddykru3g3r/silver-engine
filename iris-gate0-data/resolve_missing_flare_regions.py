#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, re, time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DER = ROOT / 'data' / 'derived'
RAW = ROOT / 'data' / 'raw' / 'solar_event_reports'
BASE = 'https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_event_reports'


def qint(x):
    try:
        v=float(x)
        return int(v) if math.isfinite(v) else None
    except Exception:
        return None


def parse_trec(s):
    m=re.match(r'(\d{4})\.(\d{2})\.(\d{2})_(\d{2}):(\d{2}):(\d{2})',str(s))
    if not m: return pd.NaT
    return pd.Timestamp(f'{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}Z')


def hhmm_to_ts(day: pd.Timestamp, token: str):
    if token is None: return pd.NaT
    t=str(token).strip().lstrip('ABU+')
    if not re.fullmatch(r'\d{4}',t): return pd.NaT
    h=int(t[:2]); m=int(t[2:])
    if h>23 or m>59: return pd.NaT
    return pd.Timestamp(day.date(),tz='UTC') + pd.Timedelta(hours=h,minutes=m)


def parse_report(text: str, day: pd.Timestamp):
    rows=[]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(('#',':')): continue
        # Edited-event records are fixed-ish width. Split after normalising the optional '+' flag.
        toks=line.split()
        if len(toks)<8 or not toks[0].isdigit(): continue
        event=toks[0]
        k=1
        if toks[k]=='+': k+=1
        if len(toks)<k+7: continue
        begin=toks[k]; peak=toks[k+1]; end=toks[k+2]
        obs=toks[k+3]; qual=toks[k+4]; typ=toks[k+5]
        tail=toks[k+6:]
        # A region number is conventionally the final integer token. Reject small integers/frequencies.
        region=None
        if tail and re.fullmatch(r'\d{4,5}',tail[-1]):
            rv=int(tail[-1])
            if rv>=1000: region=rv
        fclass=None
        if typ=='XRA':
            for tok in tail:
                if re.fullmatch(r'[ABCMX]\d+(?:\.\d+)?',tok):
                    fclass=tok; break
        rows.append({
            'event_id':event,'begin':begin,'peak':peak,'end':end,'obs':obs,'qual':qual,'type':typ,
            'flare_class':fclass,'raw_report_region':region,
            'start_ts':hhmm_to_ts(day,begin),'peak_ts':hhmm_to_ts(day,peak),'line':line,
        })
    return rows


def fetch_day(session: requests.Session, day: pd.Timestamp, timeout=45):
    ymd=day.strftime('%Y%m%d')
    url=f'{BASE}/{day:%Y}/{day:%m}/{ymd}events.txt'
    path=RAW/f'{day:%Y}'/f'{day:%m}'/f'{ymd}events.txt'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0:
        return path.read_text(errors='replace'),url,'cached',None
    last=None
    for attempt in range(4):
        try:
            r=session.get(url,timeout=timeout)
            if r.status_code==404: return None,url,'404',None
            r.raise_for_status(); path.write_text(r.text)
            return r.text,url,'downloaded',None
        except Exception as e:
            last=str(e); time.sleep(1.5*(attempt+1))
    return None,url,'failed',last


def canonicalize(raw_region, t, windows):
    raw=qint(raw_region)
    if raw is None: return None,[]
    if raw>=10000:
        return raw,[raw]
    cand=[n for n,(a,b) in windows.items() if n%10000==raw and t>=a-pd.Timedelta(days=2) and t<=b+pd.Timedelta(days=2)]
    return (cand[0] if len(cand)==1 else None),cand


def class_value(s):
    m=re.fullmatch(r'([ABCMX])(\d+(?:\.\d+)?)',str(s).strip())
    if not m: return None,None
    return m.group(1),float(m.group(2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-events',type=int,default=0)
    args=ap.parse_args()

    goes=pd.read_csv(DER/'goes_m1plus_interval.csv',low_memory=False)
    base=pd.read_csv(DER/'resolved_m1plus_events.csv',low_memory=False)
    sharp=pd.read_csv(DER/'sharp_metadata.csv.gz',dtype={'NOAA_ARS':str},low_memory=False)
    hmap=pd.read_csv(DER/'harp_noaa_mapping.csv',dtype={'noaa_ars':str},low_memory=False)

    # Build canonical NOAA activity windows from the real SHARP census and official HARP mapping.
    h2n=defaultdict(set)
    for _,r in hmap.iterrows():
        for n in re.findall(r'\d+',str(r.noaa_ars)):
            if int(n)>=1000: h2n[int(r.harpnum)].add(int(n))
    sharp['time']=sharp.T_REC.map(parse_trec)
    windows={}
    for _,r in sharp[sharp.time.notna()].iterrows():
        h=qint(r.HARPNUM)
        if h is None: continue
        for n in h2n.get(h,set()):
            if n not in windows: windows[n]=[r.time,r.time]
            else: windows[n]=[min(windows[n][0],r.time),max(windows[n][1],r.time)]

    goes['start_ts']=pd.to_datetime(goes.start_time,utc=True,errors='coerce')
    goes['peak_ts']=pd.to_datetime(goes.time,utc=True,errors='coerce')
    unresolved=base[base.canonical_noaa_ar.isna()].copy()
    if args.max_events: unresolved=unresolved.head(args.max_events)

    # Fetch every unique event-report day needed, plus adjacent days for midnight-crossing events.
    days=set()
    for t in pd.to_datetime(unresolved.event_start,utc=True,errors='coerce').dropna():
        d=t.normalize(); days.update([d-pd.Timedelta(days=1),d,d+pd.Timedelta(days=1)])
    session=requests.Session(); session.headers.update({'User-Agent':'IRIS-student-research/1.0'})
    report_rows=[]; fetch_log=[]
    for i,d in enumerate(sorted(days),1):
        text,url,status,err=fetch_day(session,d)
        fetch_log.append({'date':d.date().isoformat(),'url':url,'status':status,'error':err or ''})
        if text: report_rows.extend(parse_report(text,d))
        if i%100==0 or i==len(days): print(f'event reports {i}/{len(days)}',flush=True)
    pd.DataFrame(fetch_log).to_csv(DER/'event_report_fetch_log.csv',index=False)
    rep=pd.DataFrame(report_rows)
    if rep.empty: raise SystemExit('No edited event-report rows parsed')

    # Collapse all lines sharing an edited-event number/day. Region can be carried by an FLA/RSP line
    # even when the XRA line itself omits it.
    rep['day']=pd.to_datetime(rep.start_ts,utc=True,errors='coerce').dt.normalize()
    groups=[]
    for (day,eid),g in rep.groupby(['day','event_id'],dropna=False):
        x=g[g.type=='XRA']
        if x.empty: continue
        regions=sorted({int(v) for v in g.raw_report_region.dropna().astype(int) if int(v)>=1000})
        for _,xr in x.iterrows():
            groups.append({'day':day,'event_id':str(eid),'start_ts':xr.start_ts,'peak_ts':xr.peak_ts,
                           'flare_class':xr.flare_class,'report_regions':';'.join(map(str,regions)),
                           'n_report_regions':len(regions),'xra_line':xr.line})
    xra=pd.DataFrame(groups)

    results=[]
    for _,u in unresolved.iterrows():
        idx=int(u.event_index)
        g=goes.iloc[idx]
        t=g.start_ts; peak=g.peak_ts; gc=str(g.flare_class)
        gid=qint(g.get('event_id_swpc'))
        gl,gv=class_value(gc)
        cand=xra.copy()
        if gid is not None:
            exact=cand[cand.event_id==str(gid)]
            if not exact.empty: cand=exact
        if pd.notna(t):
            cand=cand[(cand.start_ts-t).abs()<=pd.Timedelta(minutes=35)]
        scored=[]
        for _,c in cand.iterrows():
            cl,cv=class_value(c.flare_class)
            if gl and cl and gl!=cl: continue
            sd=abs((c.start_ts-t).total_seconds())/60 if pd.notna(c.start_ts) and pd.notna(t) else 999
            pdiff=abs((c.peak_ts-peak).total_seconds())/60 if pd.notna(c.peak_ts) and pd.notna(peak) else 20
            cd=abs(math.log10(max(cv,1e-6)/max(gv,1e-6))) if cv and gv else 0.2
            event_bonus=-25 if gid is not None and c.event_id==str(gid) else 0
            score=sd+0.5*pdiff+20*cd+event_bonus
            scored.append((score,c))
        scored.sort(key=lambda z:z[0])
        chosen=None; reason='NO_REPORT_MATCH'; confidence='none'; canon=None; raw=None; cn=[]
        if scored:
            score,c=scored[0]
            # Require the best candidate to be clearly separated unless SWPC event id is exact.
            margin=(scored[1][0]-score) if len(scored)>1 else 999
            regs=[int(v) for v in str(c.report_regions).split(';') if v and v!='nan']
            if len(regs)==1:
                raw=regs[0]; canon,cn=canonicalize(raw,t,windows)
                if canon is not None:
                    exact_id=(gid is not None and c.event_id==str(gid))
                    if exact_id or (score<=20 and margin>=5):
                        chosen=c; reason='EDITED_EVENT_REPORT_EXACT_ID' if exact_id else 'EDITED_EVENT_REPORT_TIME_CLASS'
                        confidence='high' if exact_id or score<=10 else 'medium'
            best_score=float(score); best_margin=float(margin)
        else:
            best_score=None; best_margin=None
        results.append({
            'event_index':idx,'event_start':u.event_start,'goes_class':gc,
            'canonical_noaa_ar':canon if chosen is not None else None,
            'raw_report_region':raw if chosen is not None else None,
            'resolution':reason,'confidence':confidence,
            'report_event_id':chosen.event_id if chosen is not None else '',
            'score':best_score,'margin_to_second':best_margin,
            'candidate_noaas':';'.join(map(str,cn)) if chosen is not None else '',
            'xra_line':chosen.xra_line if chosen is not None else '',
        })

    out=pd.DataFrame(results)
    out.to_csv(DER/'supplemental_region_resolutions.csv',index=False)
    resolved_n=int(out.canonical_noaa_ar.notna().sum())
    summary={
        'baseline_unresolved_events':int(len(base)-base.canonical_noaa_ar.notna().sum()),
        'events_attempted':int(len(out)),
        'supplementally_resolved':resolved_n,
        'remaining_after_supplement':int((len(base)-base.canonical_noaa_ar.notna().sum())-resolved_n),
        'high_confidence':int((out.confidence=='high').sum()),
        'medium_confidence':int((out.confidence=='medium').sum()),
        'event_report_days_requested':len(days),
        'event_report_fetch_failures':int((pd.DataFrame(fetch_log).status=='failed').sum()),
        'note':'Only unique edited-event report matches with a uniquely canonicalizable NOAA region are auto-applied. Ambiguous cases remain unresolved for human audit.'
    }
    (DER/'supplemental_resolution_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
