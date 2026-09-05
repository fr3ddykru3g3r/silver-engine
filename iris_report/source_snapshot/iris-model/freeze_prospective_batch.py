from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

BANNED_TOKENS=('label','target','outcome','flare_class','goes_class','future_event','observed_event')
REQUIRED=('sample_id','region_group_id','harpnum','noaa_ars','input_t_rec','predicted_probability','decision_threshold')


def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--protocol',default='PROSPECTIVE_BLIND_VALIDATION_PROTOCOL.md')
    ap.add_argument('--commit-sha',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--issued-at-utc',default=None,help='ISO UTC; default is current UTC at freeze time')
    ap.add_argument('--max-input-latency-minutes',type=float,default=120.0)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    src=Path(args.predictions); ck=Path(args.checkpoint); protocol=Path(args.protocol)
    df=pd.read_csv(src)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f'missing required prediction fields: {missing}')
    bad=[c for c in df.columns if any(tok in c.lower() for tok in BANNED_TOKENS)]
    # decision_threshold is allowed despite the substring token list being conservative.
    bad=[c for c in bad if c!='decision_threshold']
    if bad: raise ValueError(f'outcome-like columns forbidden at freeze: {bad}')
    if not df.sample_id.astype(str).is_unique: raise ValueError('sample_id must be unique')
    if ((df.predicted_probability<0)|(df.predicted_probability>1)).any(): raise ValueError('probability outside [0,1]')
    if ((df.decision_threshold<0)|(df.decision_threshold>1)).any(): raise ValueError('threshold outside [0,1]')

    issued=datetime.now(timezone.utc) if args.issued_at_utc is None else datetime.fromisoformat(args.issued_at_utc.replace('Z','+00:00')).astimezone(timezone.utc)
    t=pd.to_datetime(df.input_t_rec,utc=True,errors='raise')
    issued_ts=pd.Timestamp(issued)
    if (t>issued_ts).any(): raise ValueError('input magnetogram time later than issuance')
    latency=(issued_ts-t).dt.total_seconds()/60.0
    if (latency<0).any() or (latency>args.max_input_latency_minutes).any():
        raise ValueError(f'input latency exceeds {args.max_input_latency_minutes} min; min={latency.min()} max={latency.max()}')

    frozen=df.copy()
    frozen['issued_at_utc']=issued.isoformat().replace('+00:00','Z')
    frozen['horizon_end_utc']=(issued+timedelta(hours=24)).isoformat().replace('+00:00','Z')
    frozen['input_latency_minutes']=latency.to_numpy()
    frozen['decision']= (frozen.predicted_probability >= frozen.decision_threshold).astype(int)
    fp=out/'prospective_predictions_frozen.csv'; frozen.to_csv(fp,index=False)
    meta={
      'status':'FROZEN_BEFORE_OUTCOME_COLLECTION',
      'issued_at_utc':issued.isoformat().replace('+00:00','Z'),
      'horizon_end_utc':(issued+timedelta(hours=24)).isoformat().replace('+00:00','Z'),
      'rows':len(frozen),'connected_regions':int(frozen.region_group_id.astype(str).nunique()),
      'checkpoint_sha256':sha256_file(ck),'protocol_sha256':sha256_file(protocol),
      'predictions_sha256':sha256_file(fp),'source_predictions_sha256':sha256_file(src),
      'code_commit_sha':args.commit_sha,'max_input_latency_minutes':args.max_input_latency_minutes,
      'outcomes_queried_by_this_script':False,
    }
    mp=out/'prospective_freeze_metadata.json'; mp.write_text(json.dumps(meta,indent=2)+'\n')
    # Hash chain makes later mutation obvious.
    chain=hashlib.sha256((meta['checkpoint_sha256']+meta['protocol_sha256']+meta['predictions_sha256']+args.commit_sha+meta['issued_at_utc']).encode()).hexdigest()
    (out/'FREEZE_SHA256.txt').write_text(chain+'\n')
    print(json.dumps(meta|{'freeze_chain_sha256':chain},indent=2))

if __name__=='__main__': main()
