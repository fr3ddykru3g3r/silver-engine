from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from cdr_models import FeatureTransformer
from cdr_data import FeatureSequenceDataset
from metrics import all_metrics, region_bootstrap

FROZEN = {
    'supervised_2': {'n_features':2,'threshold':0.26,'checkpoint':'model.pt','kind':'supervised'},
    'cdr_2': {'n_features':2,'threshold':0.46,'checkpoint':'cdr_model.pt','kind':'cdr'},
    'supervised_10': {'n_features':10,'threshold':0.33,'checkpoint':'model.pt','kind':'supervised'},
    'cdr_10': {'n_features':10,'threshold':0.41,'checkpoint':'cdr_model.pt','kind':'cdr'},
}


def predict(model, ds, batch_size=128):
    rows=[];model.eval()
    with torch.no_grad():
        for st in range(0,len(ds),batch_size):
            z=[ds[i] for i in range(st,min(st+batch_size,len(ds)))]
            x=torch.stack([a['x'] for a in z])
            p=torch.sigmoid(model(x)).cpu().numpy()
            rows.extend({'sample_id':a['sample_id'],'region_group_id':a['group'],'y':int(a['y'].item()),'p':float(pp)} for a,pp in zip(z,p))
    return pd.DataFrame(rows)


def load_model(model_dir:Path,name:str,nf:int):
    spec=FROZEN[name];ck=torch.load(model_dir/spec['checkpoint'],map_location='cpu')
    state=ck['state_dict'];seq_len=int(ck.get('seq_len',40));m=FeatureTransformer(nf,seq_len=seq_len);m.load_state_dict(state);m.eval()
    recorded=float(ck.get('threshold',spec['threshold']))
    if abs(recorded-spec['threshold'])>1e-12:
        raise RuntimeError(f'{name}: checkpoint threshold {recorded} != frozen {spec["threshold"]}')
    return m,recorded


def paired_delta(a:pd.DataFrame,b:pd.DataFrame,ta:float,tb:float,n_boot:int,seed:int):
    """Cluster-paired bootstrap; returns metric(a)-metric(b)."""
    keys=['sample_id','region_group_id','y'];m=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(m)!=len(a) or len(m)!=len(b):raise RuntimeError('paired predictions are not on identical samples')
    groups=np.asarray(sorted(m.region_group_id.unique()));rng=np.random.default_rng(seed);vals={k:[] for k in ['tss','hss','auroc','auprc','brier','bss','ece10']}
    for _ in range(n_boot):
        draw=rng.choice(groups,size=len(groups),replace=True);parts=[m[m.region_group_id.eq(g)] for g in draw];z=pd.concat(parts,ignore_index=True)
        ma=all_metrics(z.y,z.p_a,ta);mb=all_metrics(z.y,z.p_b,tb)
        for k in vals:vals[k].append(ma[k]-mb[k])
    out={}
    for k,v in vals.items():
        x=np.asarray(v,dtype=float);finite=x[np.isfinite(x)]
        out[k]={'delta_median':float(np.median(finite)),'lo95':float(np.percentile(finite,2.5)),'hi95':float(np.percentile(finite,97.5)),
                'p_two_sided':float(min(1.0,2*min(np.mean(finite<=0),np.mean(finite>=0))))}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--models-root',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--bootstrap',type=int,default=5000)
    a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);root=Path(a.models_root)
    datasets={nf:FeatureSequenceDataset(a.evidence_dir,'test',n_features=nf,per_group=4,pos_cap=1,seed=2028) for nf in [2,10]}
    reports={};preds={}
    for name,spec in FROZEN.items():
        d=root/name/'outputs'/name
        if not d.exists():d=root/name
        model,thr=load_model(d,name,spec['n_features']);pred=predict(model,datasets[spec['n_features']]);pred.to_csv(out/f'{name}_test_predictions.csv',index=False);preds[name]=pred
        rep={'model':name,'test_protocol':'one-shot locked test; no retraining; validation-frozen threshold','threshold':thr,'test_items':len(pred),'test_groups':int(pred.region_group_id.nunique()),
             'positive_items':int(pred.y.sum()),'metrics':all_metrics(pred.y,pred.p,thr),'region_bootstrap':region_bootstrap(pred,a.bootstrap,20260826,thr)}
        reports[name]=rep;(out/f'{name}_test_metrics.json').write_text(json.dumps(rep,indent=2,allow_nan=True)+'\n');print(json.dumps(rep),flush=True)
    paired={
        'cdr_2_minus_supervised_2':paired_delta(preds['cdr_2'],preds['supervised_2'],FROZEN['cdr_2']['threshold'],FROZEN['supervised_2']['threshold'],a.bootstrap,2202),
        'cdr_10_minus_supervised_10':paired_delta(preds['cdr_10'],preds['supervised_10'],FROZEN['cdr_10']['threshold'],FROZEN['supervised_10']['threshold'],a.bootstrap,2210),
    }
    final={'frozen':FROZEN,'test_dataset_seed':2028,'test_per_group':4,'test_pos_cap':1,'bootstrap_replicates':a.bootstrap,'models':reports,'paired_deltas':paired}
    (out/'cdr_locked_test_summary.json').write_text(json.dumps(final,indent=2,allow_nan=True)+'\n')

if __name__=='__main__':main()
