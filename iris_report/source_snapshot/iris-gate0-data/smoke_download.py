#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, tempfile
import argparse
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parent
DER=ROOT/'data'/'derived'


def sha256_bytes(b:bytes): return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', default=None,
                    help='Evidence root containing data/derived; defaults to the legacy local path')
    ap.add_argument('--out', default=None,
                    help='Output JSON path; defaults beside the selected URL manifest')
    args = ap.parse_args()
    evidence = Path(args.evidence_dir) if args.evidence_dir else ROOT
    der = evidence / 'data' / 'derived' if args.evidence_dir else DER
    p=der/'image_urls_all_splits.csv.gz'
    if not p.exists(): raise SystemExit('image_urls_all_splits.csv.gz missing')
    df=pd.read_csv(p)
    rows=[]
    # Deterministic end-to-end proof: first sample in each primary split.
    for split in ['train','validation','test']:
        x=df[df.partition.eq(split)].sort_values(['t_rec','sample_id'])
        if x.empty: raise SystemExit(f'empty {split} split')
        r=x.iloc[0]
        url=str(r.magnetogram_url)
        resp=requests.get(url,timeout=180,allow_redirects=True)
        resp.raise_for_status()
        data=resp.content
        if len(data)<2880: raise RuntimeError(f'{split}: retrieved object is too small for FITS: {len(data)} bytes')
        # FITS primary header begins with SIMPLE  = in the first card for these JSOC products.
        head=data[:80].decode('ascii','ignore')
        if 'SIMPLE' not in head:
            raise RuntimeError(f'{split}: response does not look like FITS; first card={head!r}')
        rows.append({
            'partition':split,
            'sample_id':str(r.sample_id),
            'harpnum':int(r.harpnum),
            't_rec':str(r.t_rec),
            'url':url,
            'http_status':int(resp.status_code),
            'bytes':len(data),
            'sha256':sha256_bytes(data),
            'fits_first_card':head.strip()
        })
        print(f'{split}: {r.sample_id} {len(data):,} bytes FITS OK',flush=True)
    out={'status':'PASS','samples':rows}
    output = Path(args.out) if args.out else der/'fits_retrieval_smoke_test.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
