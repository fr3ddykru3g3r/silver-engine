from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REQUIRED=['sample_id','region_group_id','forecast_issue_utc','forecast_horizon_hours','model_probability','model_threshold','predicted_class','model_commit_sha','checkpoint_sha256']


def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--note',default='')
    a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    p=Path(a.predictions); df=pd.read_csv(p)
    miss=[c for c in REQUIRED if c not in df.columns]
    if miss: raise RuntimeError(f'missing columns: {miss}')
    df['forecast_issue_utc']=pd.to_datetime(df.forecast_issue_utc,utc=True)
    now=pd.Timestamp.now(tz='UTC')
    if (df.forecast_issue_utc>now+pd.Timedelta(minutes=10)).any(): raise RuntimeError('future issue timestamp beyond clock tolerance')
    if (df.forecast_horizon_hours.astype(float)!=24).any(): raise RuntimeError('prospective protocol is frozen to 24 h')
    if 'observed_label' in df.columns or 'flare_outcome' in df.columns: raise RuntimeError('outcome columns forbidden at freeze time')
    frozen=out/'prospective_predictions_frozen.csv'; df.sort_values(['forecast_issue_utc','region_group_id','sample_id']).to_csv(frozen,index=False)
    digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    receipt={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'rows':len(df),'groups':int(df.region_group_id.nunique()),'prediction_sha256':digest,'rule':'Outcome labels must be appended only after the full 24 h horizon has elapsed; never overwrite this frozen file.','note':a.note}
    (out/'prospective_freeze_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
