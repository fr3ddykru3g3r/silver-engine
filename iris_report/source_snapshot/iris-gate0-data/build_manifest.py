#!/usr/bin/env python3
from __future__ import annotations
import json, hashlib, math, re
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import sys
import pandas as pd

COMMON = Path(__file__).resolve().parents[1] / 'common'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from jsoc_time import parse_jsoc_trec_to_utc

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'


def hsh(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def parse_noaas(x):
    if pd.isna(x): return set()
    return {int(n) for n in re.findall(r'\d+',str(x)) if int(n)>=1000}

def qnum(x):
    try:
        v=float(x); return v if math.isfinite(v) else None
    except: return None

def qint(x):
    v=qnum(x); return int(v) if v is not None else None

def quality_ok(x):
    if x is None or (isinstance(x,float) and math.isnan(x)): return False
    s=str(x).strip()
    try: return int(s,0)==0
    except:
        try: return int(float(s))==0
        except: return False

class UF:
    def __init__(self): self.p={}
    def add(self,x): self.p.setdefault(x,x)
    def find(self,x):
        self.add(x)
        if self.p[x]!=x:self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a


def main():
    sharp=pd.read_csv(DER/'sharp_metadata.csv.gz',dtype={'NOAA_ARS':str},low_memory=False)
    events=pd.read_csv(DER/'goes_m1plus_interval.csv',low_memory=False)
    hmap=pd.read_csv(DER/'harp_noaa_mapping.csv',dtype={'noaa_ars':str})
    # T_REC is a TAI record key.  Do not strip the suffix and relabel it UTC:
    # the resulting sub-minute shift can change an hourly metadata join.
    sharp['time']=parse_jsoc_trec_to_utc(sharp['T_REC'])
    bad=sharp.time.isna()
    if bad.any():
        raise SystemExit(f'{int(bad.sum())} SHARP rows have an invalid T_REC; refusing to build a partial manifest')
    sharp=sharp[sharp.time.notna()].copy()
    sharp['HARPNUM']=pd.to_numeric(sharp.HARPNUM,errors='coerce').astype('Int64')
    sharp=sharp[sharp.HARPNUM.notna()].copy(); sharp['HARPNUM']=sharp.HARPNUM.astype(int)

    # Lifetime union of official mapping plus record-level NOAA associations.
    h2n=defaultdict(set)
    for _,r in hmap.iterrows(): h2n[int(r.harpnum)] |= parse_noaas(r.noaa_ars)
    for _,r in sharp.iterrows(): h2n[int(r.HARPNUM)] |= parse_noaas(r.NOAA_ARS)
    uf=UF()
    for h,ns in h2n.items():
        hn=('H',h); uf.add(hn)
        for n in ns: uf.union(hn,('N',n))
    comps=defaultdict(lambda:{'harps':set(),'noaas':set()})
    for x in list(uf.p):
        root=uf.find(x); (comps[root]['harps'] if x[0]=='H' else comps[root]['noaas']).add(x[1])
    groups=[]; hgid={}; ngid={}
    for i,c in enumerate(sorted(comps.values(),key=lambda z:min(z['harps']) if z['harps'] else 10**9),1):
        gid=f'RG{i:05d}'
        for h in c['harps']: hgid[h]=gid
        for n in c['noaas']: ngid[n]=gid
        groups.append({'region_group_id':gid,'harpnums':';'.join(map(str,sorted(c['harps']))),'noaa_ars':';'.join(map(str,sorted(c['noaas']))),'n_harps':len(c['harps']),'n_noaas':len(c['noaas'])})
    pd.DataFrame(groups).to_csv(DER/'connected_region_groups.csv',index=False)

    # Determine NOAA activity windows from real SHARP records for safe 4-digit SWPC/NCEI resolution.
    nw={}
    for _,r in sharp.iterrows():
        for n in h2n.get(int(r.HARPNUM),set()):
            if n not in nw: nw[n]=[r.time,r.time]
            else: nw[n]=[min(nw[n][0],r.time),max(nw[n][1],r.time)]

    cols={c.lower():c for c in events.columns}
    def pick(*keys):
        for k in keys:
            if k in cols:return cols[k]
        for c in events.columns:
            if any(k in c.lower() for k in keys):return c
        raise KeyError(keys)
    cstart=pick('start_time'); car=pick('active_region','active region'); cclass=pick('flare_class','class')
    events['event_start']=pd.to_datetime(events[cstart],utc=True,errors='coerce')
    resolved=[]
    for idx,r in events.iterrows():
        raw=qint(r[car]); candidates=[]
        if raw is not None:
            if raw>=10000: candidates=[raw]
            else:
                candidates=[n for n,(a,b) in nw.items() if n%10000==raw and r.event_start>=a-timedelta(days=2) and r.event_start<=b+timedelta(days=2)]
        canon=candidates[0] if len(candidates)==1 else None
        resolved.append({'event_index':int(idx),'event_start':r.event_start,'goes_class':str(r[cclass]),'raw_active_region':raw,'canonical_noaa_ar':canon,'resolution':'EXACT' if raw and raw>=10000 else ('LAST4_TIME_UNIQUE' if canon else ('AMBIGUOUS' if len(candidates)>1 else 'UNRESOLVED')),'candidate_noaas':';'.join(map(str,candidates))})
    ev=pd.DataFrame(resolved)
    ev.to_csv(DER/'resolved_m1plus_events.csv',index=False)

    # Geometry/quality eligibility. SHARP LON_FWT is Stonyhurst longitude, so |LON_FWT| is CMD proxy.
    sharp['lon']=pd.to_numeric(sharp.LON_FWT,errors='coerce'); sharp['lat']=pd.to_numeric(sharp.LAT_FWT,errors='coerce')
    sharp['eligible']=sharp.QUALITY.map(quality_ok) & sharp.lon.abs().le(30) & sharp.HARPNUM.map(lambda h: bool(h2n.get(int(h))))
    sharp['group']=sharp.HARPNUM.map(hgid)
    elig=sharp[sharp.eligible & sharp.group.notna()].copy()

    spans=elig.groupby('group').time.agg(['min','max']).reset_index().sort_values(['min','group']).reset_index(drop=True)
    if len(spans)<10: raise SystemExit(f'Too few connected region groups: {len(spans)}')
    n=len(spans); i1=max(1,min(n-2,round(n*.6))); i2=max(i1+1,min(n-1,round(n*.8)))
    b1=spans.loc[i1,'min']; b2=spans.loc[i2,'min']; buf=pd.Timedelta(hours=36)
    part={}
    for _,s in spans.iterrows():
        gid=s['group']; a=s['min']; b=s['max']
        touch1=b>=b1-buf and a<=b1+buf; touch2=b>=b2-buf and a<=b2+buf
        if touch1 or touch2: p='excluded'
        elif b<b1-buf:p='train'
        elif a>b1+buf and b<b2-buf:p='validation'
        elif a>b2+buf:p='test'
        else:p='excluded'
        part[gid]=p
    if any(not any(p==x for p in part.values()) for x in ['train','validation','test']): raise SystemExit('Empty primary partition')

    # Label every eligible record from resolved attributed M1+ flare onset in (t,t+24h].
    by_noaa=defaultdict(list)
    for _,e in ev[ev.canonical_noaa_ar.notna()].iterrows(): by_noaa[int(e.canonical_noaa_ar)].append(e)
    rows=[]
    for _,r in sharp.iterrows():
        h=int(r.HARPNUM); gid=hgid.get(h); p=part.get(gid,'excluded') if bool(r.eligible) else 'excluded'
        noaas=sorted(h2n.get(h,set())); t=r.time; matches=[]
        for n in noaas:
            for e in by_noaa.get(n,[]):
                if t < e.event_start <= t+pd.Timedelta(hours=24): matches.append(e)
        label=1 if matches else (0 if bool(r.eligible) and p!='excluded' else None)
        rows.append({'sample_id':f'H{h}_{t.strftime("%Y%m%dT%H%M%SZ")}', 't_rec':t.isoformat(),'harpnum':h,'region_group_id':gid,'noaa_ars':';'.join(map(str,noaas)),'latitude_deg':qnum(r.LAT_FWT),'cmd_deg':qnum(r.LON_FWT),'quality':str(r.QUALITY),'partition':p,'label_m1plus_24h':label,'matched_event_indices':';'.join(str(int(x.event_index)) for x in matches),'max_goes_class':max([str(x.goes_class) for x in matches],default=''),'magnetogram_url':r.get('segment_magnetogram'),'usflux':qnum(r.get('USFLUX')),'r_value':qnum(r.get('R_VALUE'))})
    man=pd.DataFrame(rows)
    man.to_csv(DER/'training_manifest.csv.gz',index=False,compression='gzip')

    prim=man[man.partition.isin(['train','validation','test']) & man.label_m1plus_24h.notna()].copy()
    # Training cannot proceed if any primary split record lacks the actual image locator.
    missing_url=prim.magnetogram_url.isna() | prim.magnetogram_url.astype(str).str.strip().isin(['','nan','None'])
    if missing_url.any():
        bad=prim.loc[missing_url,['sample_id','partition','harpnum','t_rec']]
        bad.to_csv(DER/'missing_primary_image_urls.csv',index=False)
        raise SystemExit(f'{len(bad)} primary split rows have no magnetogram URL; refusing training-ready status')

    counts=[]
    for p in ['train','validation','test']:
        x=prim[prim.partition==p]; pos=x[x.label_m1plus_24h==1]
        counts.append({'partition':p,'rows':len(x),'positive_rows':len(pos),'independent_groups':x.region_group_id.nunique(),'independent_positive_groups':pos.region_group_id.nunique(),'independent_harps':x.harpnum.nunique(),'independent_positive_harps':pos.harpnum.nunique(),'image_urls':int(x.magnetogram_url.notna().sum())})
    pd.DataFrame(counts).to_csv(DER/'independent_positive_region_counts.csv',index=False)

    split={'method':'connected HARP-NOAA groups, chronology 60/20/20, 36h buffer','boundary_1':b1.isoformat(),'boundary_2':b2.isoformat(),'parts':part}
    split['sha256']=hsh(split); (DER/'frozen_split.json').write_text(json.dumps(split,indent=2,sort_keys=True)+'\n')
    audit={'total_sharp_rows':len(sharp),'interval_harps':int(sharp.HARPNUM.nunique()),'eligible_rows':int(sharp.eligible.sum()),'active_connected_groups':int(elig.group.nunique()),'all_mapping_connected_groups':len(groups),'resolved_m1plus_events':int(ev.canonical_noaa_ar.notna().sum()),'unresolved_or_ambiguous_m1plus_events':int(ev.canonical_noaa_ar.isna().sum()),'primary_rows':int(len(prim)),'primary_image_urls':int(prim.magnetogram_url.notna().sum()),'missing_primary_image_urls':int(missing_url.sum()),'time_scale':'T_REC parsed as TAI and converted to UTC with Astropy','partitions':counts,'split_sha256':split['sha256']}
    (DER/'manifest_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')

    # Complete binary acquisition plans. Generator training must use train only;
    # downstream calibration/evaluation also require validation and test images.
    cols_out=['sample_id','magnetogram_url','label_m1plus_24h','partition','region_group_id','harpnum','t_rec','noaa_ars','cmd_deg']
    prim[cols_out].to_csv(DER/'image_urls_all_splits.csv.gz',index=False,compression='gzip')
    prim[prim.partition=='train'][cols_out].to_csv(DER/'training_image_urls.csv.gz',index=False,compression='gzip')
    prim[prim.partition=='validation'][cols_out].to_csv(DER/'validation_image_urls.csv.gz',index=False,compression='gzip')
    prim[prim.partition=='test'][cols_out].to_csv(DER/'test_image_urls.csv.gz',index=False,compression='gzip')
    print(json.dumps(audit,indent=2,default=str))

if __name__=='__main__': main()
