from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from cdr_models import CDRImageBiLSTM
from cdr_data import cache_native_image_sequences
from metrics import all_metrics, region_bootstrap


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def predict(model, ds, device, batch_size=16):
    model.eval(); rows=[]
    for st in range(0,len(ds),batch_size):
        items=[ds[i] for i in range(st,min(st+batch_size,len(ds)))]
        x=torch.stack([z['x'] for z in items]).to(device)
        p=torch.sigmoid(model(x)).cpu().numpy()
        rows.extend({'sample_id':z['sample_id'],'region_group_id':z['group'],'y':int(z['y'].item()),'p':float(pp)} for z,pp in zip(items,p))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out-dir',required=True); ap.add_argument('--cache-dir',required=True)
    ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--download-workers',type=int,default=16)
    args=ap.parse_args(); seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    # Exact test construction predeclared by the validation code: test is the third partition,
    # therefore deterministic selector seed = training seed + 2, with one 40x36m sequence/group.
    ds,end,frames=cache_native_image_sequences(args.evidence_dir,'test',Path(args.cache_dir),
                                                per_group=1,pos_cap=1,seed=args.seed+2,
                                                workers=args.download_workers)
    ck=torch.load(args.checkpoint,map_location='cpu')
    state=ck.get('state_dict',ck)
    threshold=float(ck['threshold'])
    model=CDRImageBiLSTM(); model.load_state_dict(state)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=model.to(device)
    pred=predict(model,ds,device); pred.to_csv(out/'test_predictions.csv',index=False)
    met=all_metrics(pred.y.values,pred.p.values,threshold)
    boot=region_bootstrap(pred,5000,args.seed,threshold)
    report={'protocol':'one-shot frozen checkpoint; no retraining; validation-frozen threshold',
            'seed':args.seed,'test_selector_seed':args.seed+2,'threshold':threshold,
            'test_items':len(pred),'test_groups':int(pred.region_group_id.nunique()),
            'test_positive_items':int(pred.y.sum()),'metrics':met,'region_bootstrap_5000':boot}
    (out/'test_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    print(json.dumps(report,indent=2,allow_nan=True),flush=True)

if __name__=='__main__': main()
