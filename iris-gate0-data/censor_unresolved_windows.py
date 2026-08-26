#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'

man_path=DER/'training_manifest.csv.gz'
ev_path=DER/'resolved_m1plus_events.csv'

man=pd.read_csv(man_path,low_memory=False)
ev=pd.read_csv(ev_path,low_memory=False)
man['t']=pd.to_datetime(man.t_rec,utc=True,errors='coerce')
ev['event_start_ts']=pd.to_datetime(ev.event_start,utc=True,errors='coerce')
unres=ev[ev.canonical_noaa_ar.isna() & ev.event_start_ts.notna()].copy().sort_values('event_start_ts')

# A negative example is unsafe if any unattributed >=M1 flare begins in its future
# 24-hour forecast window. Because the source AR is unknown, the conservative rule
# censors every otherwise-negative AR observation in that global window. Resolved
# positives remain positive; the chronological connected-region split is unchanged.
mask=pd.Series(False,index=man.index)
for t in unres.event_start_ts:
    mask |= man.label_m1plus_24h.eq(0) & man.t.lt(t) & man.t.ge(t-pd.Timedelta(hours=24))

man['label_integrity_status']='RESOLVED_OR_CLEAN'
man.loc[mask,'label_integrity_status']='CENSORED_UNRESOLVED_GLOBAL'
man.loc[mask,'label_m1plus_24h']=pd.NA
man.drop(columns=['t']).to_csv(man_path,index=False,compression='gzip')

prim=man[man.partition.isin(['train','validation','test']) & man.label_m1plus_24h.notna()].copy()
counts=[]
for p in ['train','validation','test']:
    x=prim[prim.partition.eq(p)]; pos=x[x.label_m1plus_24h.eq(1)]
    counts.append({
        'partition':p,'rows':len(x),'positive_rows':len(pos),
        'independent_groups':x.region_group_id.nunique(),
        'independent_positive_groups':pos.region_group_id.nunique(),
        'independent_harps':x.harpnum.nunique(),
        'independent_positive_harps':pos.harpnum.nunique(),
        'image_urls':int(x.magnetogram_url.notna().sum()),
        'censored_negative_rows':int((mask & man.partition.eq(p)).sum()),
    })
pd.DataFrame(counts).to_csv(DER/'independent_positive_region_counts.csv',index=False)

cols_out=['sample_id','magnetogram_url','label_m1plus_24h','partition','region_group_id','harpnum','t_rec','noaa_ars','cmd_deg']
prim[cols_out].to_csv(DER/'image_urls_all_splits.csv.gz',index=False,compression='gzip')
prim[prim.partition.eq('train')][cols_out].to_csv(DER/'training_image_urls.csv.gz',index=False,compression='gzip')
prim[prim.partition.eq('validation')][cols_out].to_csv(DER/'validation_image_urls.csv.gz',index=False,compression='gzip')
prim[prim.partition.eq('test')][cols_out].to_csv(DER/'test_image_urls.csv.gz',index=False,compression='gzip')

# Keep original audit plus an explicit label-integrity addendum.
audit_path=DER/'manifest_audit.json'
audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
integrity={
    'rule':'global censor of otherwise-negative samples with an unresolved >=M1 flare onset in (t,t+24h]',
    'unresolved_events':int(len(unres)),
    'rows_censored_total':int(mask.sum()),
    'primary_rows_after_censoring':int(len(prim)),
    'partitions_after_censoring':counts,
    'split_sha256':audit.get('split_sha256'),
    'reason':'prevents unattributed major flares from being silently converted into false-negative training/evaluation labels',
}
(DER/'label_integrity_audit.json').write_text(json.dumps(integrity,indent=2)+'\n')
print(json.dumps(integrity,indent=2),flush=True)
