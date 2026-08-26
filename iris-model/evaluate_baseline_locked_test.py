from __future__ import annotations

import argparse,json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import build_records,cache_records,MagnetogramDataset
from benchmark_forecaster import group_balanced_subset
from forecaster import FlareCNN,parameter_count
from metrics import all_metrics,region_bootstrap

FROZEN_THRESHOLD=0.40
FROZEN_WIDTH=48
FROZEN_DROPOUT=0.20
FROZEN_SEED=2026


def collate(batch):
    return {'x':torch.stack([z['x'] for z in batch]),'y':torch.stack([z['y'] for z in batch]),'group':[z['group'] for z in batch],'sample_id':[z['sample_id'] for z in batch]}

@torch.no_grad()
def predict(model,loader,device):
    model.eval();rows=[]
    for b in loader:
        p=torch.sigmoid(model(b['x'].to(device))).cpu().numpy();y=b['y'].numpy()
        rows.extend({'sample_id':sid,'region_group_id':g,'y':int(yy),'p':float(pp)} for sid,g,yy,pp in zip(b['sample_id'],b['group'],y,p))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--cache-dir',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--bootstrap',type=int,default=5000);ap.add_argument('--download-workers',type=int,default=8)
    a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location='cpu');cfg=ck.get('config',{})
    if int(cfg.get('width',FROZEN_WIDTH))!=FROZEN_WIDTH or abs(float(cfg.get('dropout',FROZEN_DROPOUT))-FROZEN_DROPOUT)>1e-12:raise RuntimeError(f'checkpoint architecture differs from frozen protocol: {cfg}')
    if int(ck.get('seed',FROZEN_SEED))!=FROZEN_SEED:raise RuntimeError(f'checkpoint seed differs from frozen protocol: {ck.get("seed")}')
    if abs(float(ck.get('threshold',FROZEN_THRESHOLD))-FROZEN_THRESHOLD)>1e-12:raise RuntimeError(f'checkpoint threshold {ck.get("threshold")} != frozen {FROZEN_THRESHOLD}')
    test=build_records(a.evidence_dir,'test');test=group_balanced_subset(test,10,4,2028)
    test=cache_records(test,a.cache_dir,a.download_workers);test.to_csv(out/'test_records.csv.gz',index=False,compression='gzip')
    ds=MagnetogramDataset(test);loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,collate_fn=collate)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=FlareCNN(width=FROZEN_WIDTH,dropout=FROZEN_DROPOUT).to(device);model.load_state_dict(ck['state_dict'])
    pred=predict(model,loader,device);pred.to_csv(out/'test_predictions.csv',index=False)
    rep={'model':'w48_focal','source_checkpoint':str(a.checkpoint),'seed':FROZEN_SEED,'device':str(device),'parameters':parameter_count(model),'threshold':FROZEN_THRESHOLD,'test_protocol':'one-shot; validation-matched group sampling; no retraining','test_rows':len(pred),'test_groups':int(pred.region_group_id.nunique()),'positive_rows':int(pred.y.sum()),'metrics':all_metrics(pred.y,pred.p,FROZEN_THRESHOLD),'region_bootstrap':region_bootstrap(pred,a.bootstrap,20260826,FROZEN_THRESHOLD)}
    (out/'baseline_locked_test_metrics.json').write_text(json.dumps(rep,indent=2,allow_nan=True)+'\n');print(json.dumps(rep,indent=2,allow_nan=True),flush=True)

if __name__=='__main__':main()
