"""Independent verifier for an episode-benchmark artifact directory.

Deliberately does not import the benchmark runner. Recomputes hashes, attrition,
episode weights and persisted point metrics from CSV/NPZ evidence only.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

WEIGHTS={
 'WINDOW_OCCURRENCE_STANDARD':'standard_window_weight',
 'EPISODE_NORMALIZED_OCCURRENCE':'episode_normalized_occurrence_weight',
 'NEW_ONSET_CAUSAL':'new_onset_weight',
 'EPISODE_NORMALIZED_ONSET':'episode_normalized_onset_weight',
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def metric(y,p,t,w):
 a=p>=t; tp=float(w[(y==1)&a].sum()); fn=float(w[(y==1)&~a].sum()); fp=float(w[(y==0)&a].sum()); tn=float(w[(y==0)&~a].sum())
 pod=tp/(tp+fn) if tp+fn else np.nan; fpr=fp/(fp+tn) if fp+tn else np.nan; far=fp/(tp+fp) if tp+fp else np.nan
 den=(tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)
 return {'tp':tp,'fn':fn,'fp':fp,'tn':tn,'pod':pod,'fpr':fpr,'far':far,'tss':pod-fpr,'hss':2*(tp*tn-fn*fp)/den if den else np.nan,'brier':float(np.sum(w*(p-y)**2)/w.sum()) if w.sum() else np.nan}

def view(df,v):
 if v.startswith('WINDOW_') or v=='EPISODE_NORMALIZED_OCCURRENCE': use=df.copy(); y=use.standard_occurrence_label.astype(int).to_numpy()
 else: use=df[df.eligibility_code.isin(['ELIGIBLE_ONSET_POSITIVE','ELIGIBLE_ONSET_NEGATIVE'])].copy(); y=use.onset_label.astype(int).to_numpy()
 return use,y,use[WEIGHTS[v]].astype(float).to_numpy()

def check_predictions(df):
 probs=[]
 if df.duplicated(['fold','issue_timestamp','model']).any(): probs.append('duplicate prediction row')
 if ((df.probability<0)|(df.probability>1)|~np.isfinite(df.probability)).any(): probs.append('invalid probability')
 sig=[tuple(zip(g.fold.astype(str),g.issue_timestamp.astype(str))) for _,g in df.groupby('model')]
 if len(set(sig))!=1: probs.append('model cohorts differ')
 base=df[df.model==sorted(df.model.unique())[0]]
 occ=base[(base.standard_occurrence_label==1)&base.occurrence_episode_id.notna()].groupby('occurrence_episode_id').episode_normalized_occurrence_weight.sum()
 if len(occ) and not np.allclose(occ,1,atol=1e-12,rtol=0): probs.append('occurrence episode weights != 1')
 ons=base[base.eligibility_code=='ELIGIBLE_ONSET_POSITIVE'].groupby('onset_episode_id').episode_normalized_onset_weight.sum()
 if len(ons) and not np.allclose(ons,1,atol=1e-12,rtol=0): probs.append('onset episode weights != 1')
 if np.any(base.loc[base.eligibility_code=='ALREADY_ACTIVE_PERSISTENCE','new_onset_weight'].astype(float)!=0): probs.append('persistence carries onset weight')
 return probs

def check_results(pred,reported):
 problems=[]; n=0
 for _,r in reported.iterrows():
  v=str(r['view']); m=str(r['model'])
  if v not in WEIGHTS: continue
  use,y,w=view(pred[pred.model==m],v); p=use.probability.astype(float).to_numpy(); th=np.unique(use.threshold.astype(float))
  if len(th)!=1: problems.append(f'{m}/{v}: threshold not unique'); continue
  z=metric(y,p,float(th[0]),w)
  for k in ['tp','fn','fp','tn','pod','fpr','far','tss','hss','brier']:
   if k in r.index and not np.isclose(float(r[k]),z[k],atol=1e-10,rtol=1e-9,equal_nan=True): problems.append(f'{m}/{v}/{k}: {r[k]} != {z[k]}')
  n+=1
 return n,problems

def check_draw(path):
 z=np.load(path,allow_pickle=False); pi=z['positive_indices']; ni=z['negative_indices']; pu=z['positive_units']; nu=z['negative_units']; p=[]
 if pi.ndim!=2 or ni.ndim!=2 or pi.shape[0]!=ni.shape[0]: p.append('bad bootstrap dimensions')
 if pi.shape[1]!=len(pu) or ni.shape[1]!=len(nu): p.append('bootstrap widths mismatch')
 if len(pu) and (pi.min()<0 or pi.max()>=len(pu)): p.append('positive index out of range')
 if len(nu) and (ni.min()<0 or ni.max()>=len(nu)): p.append('negative index out of range')
 return {'passed':not p,'replicates':int(pi.shape[0]),'positive_units':len(pu),'negative_units':len(nu),'sha256':sha(path),'problems':p}

def verify(root):
 root=Path(root); problems=[]; checks={}
 manifest=json.loads((root/'evidence_hashes.json').read_text()); bad=[]
 for rel,expected in manifest.items():
  p=root/rel
  if not p.exists() or sha(p)!=expected: bad.append(rel)
 checks['hashes']={'passed':not bad,'mismatches':bad}
 led=pd.read_csv(root/'candidate_attrition_ledger.csv'); att=json.loads((root/'attrition_summary.json').read_text()); actual=dict(sorted(Counter(led.eligibility_code.astype(str)).items())); ap=[]
 if len(led)!=int(att['total_candidate_issues']): ap.append('total mismatch')
 if actual!={str(k):int(v) for k,v in att['terminal_codes'].items()}: ap.append('terminal-code mismatch')
 checks['attrition']={'passed':not ap,'rows':len(led),'terminal_codes':actual,'problems':ap}
 for tag in ['strict_2017','expanding_oof_2014_2017']:
  pred=pd.read_csv(root/f'predictions_{tag}.csv'); res=pd.read_csv(root/f'results_{tag}.csv'); pp=check_predictions(pred); n,rp=check_results(pred,res)
  checks[f'{tag}_predictions']={'passed':not pp,'problems':pp,'rows':len(pred)}
  checks[f'{tag}_metrics']={'passed':not rp,'problems':rp,'reported_rows_recomputed':n}
  draw='strict_2017_shared_bootstrap_draws.npz' if tag=='strict_2017' else 'expanding_oof_shared_bootstrap_draws.npz'
  checks[f'{tag}_draws']=check_draw(root/draw)
 passed=all(x['passed'] for x in checks.values()); return {'format':'IRIS_EPISODE_DEVELOPMENT_ARTIFACT_INDEPENDENT_VERIFICATION_V1','passed':passed,'checks':checks}

def main():
 p=argparse.ArgumentParser(); p.add_argument('artifact_dir'); p.add_argument('--output'); a=p.parse_args(); report=verify(a.artifact_dir); text=json.dumps(report,indent=2,sort_keys=True)+'\n'; print(text,end='')
 if a.output: Path(a.output).write_text(text)
 return 0 if report['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
