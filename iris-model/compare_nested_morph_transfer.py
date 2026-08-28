from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from metrics import all_metrics

ARMS=['real','duplicate','base','pil','pil_blur','geometry_flip','block_shuffle']

def load(d):
    p=pd.read_csv(d/'test_predictions.csv'); m=json.loads((d/'metrics.json').read_text()); return p,float(m['validation_threshold']),m

def delta_df(a,b):
    keys=['sample_id','region_group_id','y']; z=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(z)!=len(a) or len(z)!=len(b): raise RuntimeError('identity mismatch')
    return z

def metric_delta(z,ta,tb,k): return float(all_metrics(z.y,z.p_a,ta)[k]-all_metrics(z.y,z.p_b,tb)[k])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--seed',type=int,default=280826);a=ap.parse_args()
    root=Path(a.root);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True); strata={}
    for fd in sorted(root.glob('fold_*')):
      f=int(fd.name.split('_')[-1])
      for sd in sorted(fd.glob('seed_*')):
        s=int(sd.name.split('_')[-1]); data={x:load(sd/x) for x in ARMS if (sd/x/'test_predictions.csv').exists()}
        if set(ARMS)-set(data): raise RuntimeError(f'missing arms {fd}/{sd}: {set(ARMS)-set(data)}')
        ids=set(data['real'][0].sample_id.astype(str))
        for arm,(p,_,_) in data.items():
          if set(p.sample_id.astype(str))!=ids: raise RuntimeError(f'outer identities differ {f}/{s}/{arm}')
        strata[(f,s)]=data
    if len(strata)<8: raise RuntimeError(f'expected 4 folds x 2 seeds, got {len(strata)}')
    pairs=[('duplicate','real'),('base','real'),('pil','real'),('base','duplicate'),('pil','duplicate'),('pil','base'),('pil','pil_blur'),('pil','geometry_flip'),('base','block_shuffle')]
    metrics=['tss','hss','recall','fpr','precision','auroc','auprc','brier','bss']; rng=np.random.default_rng(a.seed); report={'protocol':'nested rolling-origin, two replicate seeds, connected-region paired bootstrap','strata':len(strata),'comparisons':{},'outer_evaluation_used_for_selection':False}
    for aa,bb in pairs:
      report['comparisons'][f'{aa}_minus_{bb}']={}
      zs=[]; points=[]
      for key,d in strata.items():
        pa,ta,_=d[aa];pb,tb,_=d[bb];z=delta_df(pa,pb);zs.append((key,z,ta,tb))
      for k in metrics:
        pts=[metric_delta(z,ta,tb,k) for _,z,ta,tb in zs]; boots=[]
        for _ in range(a.bootstrap):
          vals=[]
          for _,z,ta,tb in zs:
            groups=np.asarray(sorted(z.region_group_id.astype(str).unique())); draw=rng.choice(groups,len(groups),replace=True)
            q=pd.concat([z[z.region_group_id.astype(str).eq(g)] for g in draw],ignore_index=True);vals.append(metric_delta(q,ta,tb,k))
          boots.append(float(np.mean(vals)))
        x=np.asarray(boots);report['comparisons'][f'{aa}_minus_{bb}'][k]={'equal_stratum_mean_delta':float(np.mean(pts)),'lo95':float(np.percentile(x,2.5)),'hi95':float(np.percentile(x,97.5)),'p_two_sided':float(min(1,2*min(np.mean(x<=0),np.mean(x>=0)))),'per_stratum':{f'f{q[0]}_s{q[1]}':float(v) for q,v in zip([x[0] for x in zs],pts)},'bootstrap_replicates':a.bootstrap,'resampling_unit':'connected region within fold/seed stratum'}
    # Fidelity-to-utility: physical PIL distance versus TSS utility relative to duplicate.
    fu=[]
    for (f,s),d in strata.items():
      for arm in ['base','pil','pil_blur','geometry_flip','block_shuffle']:
        pp=root/f'fold_{f}'/f'seed_{s}'/'physical'/arm/'v2_manipulation_metrics.json'
        if not pp.exists(): continue
        ph=json.loads(pp.read_text());z=delta_df(d[arm][0],d['duplicate'][0]);dt=metric_delta(z,d[arm][1],d['duplicate'][1],'tss')
        fu.append({'fold':f,'seed':s,'arm':arm,'pil_distance':float(ph['hard_pil_standardized_energy_distance']),'generic_distance_ratio':float(ph['generic_distance_to_real_baseline_ratio']),'tss_delta_vs_duplicate':dt})
    pd.DataFrame(fu).to_csv(out/'fidelity_utility_points.csv',index=False)
    if len(fu)>=8:
      r,p=spearmanr([x['pil_distance'] for x in fu],[x['tss_delta_vs_duplicate'] for x in fu]);report['fidelity_to_utility']={'spearman_pil_distance_vs_tss_delta':float(r),'p_value':float(p),'n':len(fu),'interpretation':'negative rho means lower PIL distance tends to greater forecast utility'}
    (out/'nested_transfer_report.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');print(json.dumps(report,indent=2,allow_nan=True))
if __name__=='__main__':main()
