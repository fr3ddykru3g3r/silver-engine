#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'

goes_path=DER/'goes_m1plus_interval.csv'
supp_path=DER/'supplemental_region_resolutions.csv'
backup=DER/'goes_m1plus_interval_pre_supplement.csv'

goes=pd.read_csv(goes_path,low_memory=False)
supp=pd.read_csv(supp_path,low_memory=False)
if not backup.exists(): goes.to_csv(backup,index=False)

applied=[]
for _,r in supp[supp.canonical_noaa_ar.notna()].iterrows():
    i=int(r.event_index)
    if i<0 or i>=len(goes): continue
    before=goes.at[i,'active_region'] if 'active_region' in goes.columns else None
    if pd.isna(before):
        goes.at[i,'active_region']=int(r.canonical_noaa_ar)
        applied.append({'event_index':i,'before':None,'after':int(r.canonical_noaa_ar),'resolution':r.resolution,'confidence':r.confidence})

goes.to_csv(goes_path,index=False)
pd.DataFrame(applied).to_csv(DER/'supplemental_region_applications.csv',index=False)
summary={'supplemental_candidates':int(supp.canonical_noaa_ar.notna().sum()),'applied_to_missing_goes_regions':len(applied),'preserved_existing_regions':int(supp.canonical_noaa_ar.notna().sum())-len(applied)}
(DER/'supplemental_application_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
