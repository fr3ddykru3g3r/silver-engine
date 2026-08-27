from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from evaluate_generator_v2 import load_real, load_synthetic, pooled_vectors


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--synthetic-manifest',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--real-per-group',type=int,default=2); ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    real,_,_=load_real(args.evidence_dir,args.cache_dir,args.real_per_group,args.seed)
    syn,_,_=load_synthetic(args.synthetic_manifest)
    rp=pooled_vectors(real); sp=pooled_vectors(syn)
    nr=min(512,len(rp)); ns=min(256,len(sp)); rp=rp[:nr]; sp=sp[:ns]
    rr=torch.cdist(rp,rp); rr.fill_diagonal_(float('inf'))
    real_nn=rr.min(1).values
    syn_nn=torch.cdist(sp,rp).min(1).values
    real_med=float(real_nn.median()); syn_med=float(syn_nn.median()); ratio=syn_med/max(real_med,1e-9)
    report={'definition':'coarse 16x16 asinh pooled nearest-neighbour novelty; train-only','real_real_nn_median':real_med,'synthetic_real_nn_median':syn_med,'novelty_ratio_to_real_nn_median':ratio,'gate_definition':{'min_ratio':0.25,'max_ratio':2.5},'novelty_gate_pass':bool(0.25<=ratio<=2.5),'forecast_outcomes_accessed':False}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
