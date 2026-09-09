"""Direct execution of frozen IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1.

Scientific choices are defined by the four frozen JSON contracts dated 2026-09-09.
This file replaces only the failed base64 transport. The previous workflow failed
before data access, so no labels or model scores informed this implementation.
"""
from __future__ import annotations

import argparse, hashlib, json, math, re
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'config'
PRE=CFG/'freshness_crossover_study_v1_preregistration_2026-09-09.json'
IMP=CFG/'freshness_crossover_study_v1_implementation_freeze_2026-09-09.json'
POOL=CFG/'freshness_crossover_study_v1_policy_pooling_clarification_2026-09-09.json'
CLAR=CFG/'freshness_crossover_study_v1_statistical_and_control_clarification_2026-09-09.json'
UA='IRIS-SEP-freshness-crossover-v1/1.0'
OMNI='https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc'
XDIR='https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/{year}/{month}/goes15/netcdf/'
DELAYS=[0,5,15,30,60,120,360,720,1440]
NONZERO=DELAYS[1:]
SEEDS=[7,13,26,42,73]
XRS_THRESH=[1e-6,1e-5,1e-4]; P_THRESH=[1.,5.,8.]
ROLE_BOUNDS={
 'fit':(pd.Timestamp('2011-01-01',tz='UTC'),pd.Timestamp('2015-01-01',tz='UTC')),
 'calibration':(pd.Timestamp('2015-01-01',tz='UTC'),pd.Timestamp('2016-01-01',tz='UTC')),
 'threshold':(pd.Timestamp('2016-01-01',tz='UTC'),pd.Timestamp('2017-01-01',tz='UTC')),
 'score':(pd.Timestamp('2017-01-01',tz='UTC'),pd.Timestamp('2017-12-01',tz='UTC')),
}

def sha(b): return hashlib.sha256(b).hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')
def fetch(s,url,max_bytes=60_000_000):
 r=s.get(url,timeout=90,headers={'User-Agent':UA}); r.raise_for_status(); b=r.content
 if not b or len(b)>max_bytes: raise RuntimeError(f'bad download {url}: {len(b)} bytes')
 return b

def acquire(out:Path):
 cache=Path('/tmp/freshness_raw_cache'); cache.mkdir(parents=True,exist_ok=True); s=requests.Session(); manifest=[]
 pro=[]
 for y in range(2011,2018):
  url=OMNI.format(year=y); b=fetch(s,url); manifest.append({'family':'proton','url':url,'bytes':len(b),'sha256':sha(b)})
  rows=[]
  for line in b.decode('ascii').splitlines():
   f=line.split();
   if len(f)<7: continue
   yy,doy,hh,mm=map(int,f[:4]); v=float(f[-3]); t=pd.Timestamp(f'{yy}-01-01',tz='UTC')+pd.Timedelta(days=doy-1,hours=hh,minutes=mm)
   if v==99999.99 or v<0 or not np.isfinite(v): v=np.nan
   rows.append((t,v))
  pro.append(pd.DataFrame(rows,columns=['time','proton']))
 proton=pd.concat(pro,ignore_index=True).sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
 xparts=[]
 for y in range(2011,2018):
  maxm=11 if y==2017 else 12
  for m in range(1,maxm+1):
   d=XDIR.format(year=y,month=f'{m:02d}'); html=fetch(s,d,3_000_000).decode('utf-8','replace')
   links=re.findall(r'href=["\']([^"\']+)["\']',html,re.I)
   cand=sorted({x for x in links if re.search(r'g15_xrs_1m_.*\.nc$',Path(x).name,re.I) and 'science' not in x.lower()})
   if len(cand)!=1: raise RuntimeError(f'expected one operational XRS file {d}, got {cand}')
   url=urljoin(d,cand[0]); b=fetch(s,url); manifest.append({'family':'xrs','url':url,'bytes':len(b),'sha256':sha(b)})
   p=cache/Path(cand[0]).name; p.write_bytes(b)
   with Dataset(p) as ds:
    tt=np.asarray(ds['time_tag'][:]).astype('float64'); A=np.ma.filled(ds['A_AVG'][:],np.nan).astype(float); B=np.ma.filled(ds['B_AVG'][:],np.nan).astype(float)
    aq=np.ma.filled(ds['A_QUAL_FLAG'][:],1); bq=np.ma.filled(ds['B_QUAL_FLAG'][:],1)
   t=pd.to_datetime(tt,unit='ms',utc=True); A=A/0.85; B=B/0.7
   A[(aq!=0)|(~np.isfinite(A))|(A<0)]=np.nan; B[(bq!=0)|(~np.isfinite(B))|(B<0)]=np.nan
   xparts.append(pd.DataFrame({'time':t,'A':A,'B':B}))
 xrs=pd.concat(xparts,ignore_index=True).sort_values('time').drop_duplicates('time',keep='last').reset_index(drop=True)
 dump(out/'source_manifest.json',manifest)
 return proton,xrs

def idx(times,issue,delay):
 ns=times.view('int64'); ins=issue.value; lo=ins-int(24*3600*1e9); cutoff=min(ins-1,ins-int(delay*60*1e9))
 return int(np.searchsorted(ns,lo,'left')),int(np.searchsorted(ns,cutoff,'right'))

def stream_features(tns,v,issue_ns,expected,prefix,log1p=False):
 good=np.isfinite(v); z=v[good]; tt=tns[good]; out={f'{prefix}_n':float(len(z)),f'{prefix}_coverage':float(len(z)/expected)}
 keys=['min','max','mean','median','std','last','age','max_pos_diff','max_pos_log_change']
 for k in keys: out[f'{prefix}_{k}']=np.nan
 if len(z)==0: return out
 out.update({f'{prefix}_min':float(np.min(z)),f'{prefix}_max':float(np.max(z)),f'{prefix}_mean':float(np.mean(z)),f'{prefix}_median':float(np.median(z)),f'{prefix}_std':float(np.std(z,ddof=0)),f'{prefix}_last':float(z[-1]),f'{prefix}_age':float((issue_ns-tt[-1])/60e9)})
 if len(z)>1:
  d=np.diff(z); out[f'{prefix}_max_pos_diff']=float(max(0.,np.max(d)))
  if log1p: ld=np.diff(np.log1p(np.maximum(z,0)))
  else:
   ok=(z[:-1]>0)&(z[1:]>0); ld=np.log(z[1:][ok])-np.log(z[:-1][ok]) if np.any(ok) else np.array([])
  out[f'{prefix}_max_pos_log_change']=float(max(0.,np.max(ld))) if len(ld) else 0.
 return out

def family_features(df,issue,delay,family):
 a,b=idx(df['time'].to_numpy(),issue,delay); sub=df.iloc[a:b]; tns=sub['time'].astype('int64').to_numpy(); ins=issue.value
 if family=='proton':
  v=sub['proton'].to_numpy(float); out=stream_features(tns,v,ins,288,'p',True); good=v[np.isfinite(v)]
  for q in P_THRESH: out[f'p_count_ge_{q:g}']=float(np.sum(good>=q))
  return out
 out=stream_features(tns,sub['A'].to_numpy(float),ins,1440,'a'); out.update(stream_features(tns,sub['B'].to_numpy(float),ins,1440,'b'))
 good=sub['B'].to_numpy(float); good=good[np.isfinite(good)]
 for q in XRS_THRESH: out[f'b_count_ge_{q:.0e}']=float(np.sum(good>=q))
 return out

def label_issue(proton,issue):
 t=proton['time'].astype('int64').to_numpy(); v=proton['proton'].to_numpy(float); ins=issue.value; j=np.searchsorted(t,ins,'right')-1
 if j<0 or not np.isfinite(v[j]) or ins-t[j]>5*60e9: return {'resolved':False,'active':False,'label':None,'crossing_time':None,'reason':'issue_support'}
 if v[j]>=10: return {'resolved':True,'active':True,'label':None,'crossing_time':None,'reason':'already_active'}
 end=ins+24*3600*1e9; k=np.searchsorted(t,end,'left')
 if k>=len(t) or t[k]!=end or not np.isfinite(v[k]): return {'resolved':False,'active':False,'label':None,'crossing_time':None,'reason':'endpoint'}
 segt=t[j:k+1]; segv=v[j:k+1]
 if np.any(~np.isfinite(segv)) or np.any(np.diff(segt)>5*60e9): return {'resolved':False,'active':False,'label':None,'crossing_time':None,'reason':'gap_or_invalid'}
 cross=None
 for q in range(1,len(segv)):
  if segt[q]>ins and segv[q-1]<10<=segv[q]: cross=pd.Timestamp(segt[q],tz='UTC'); break
 return {'resolved':True,'active':False,'label':int(cross is not None),'crossing_time':cross,'reason':'ok'}

def role_for(issue):
 for r,(a,b) in ROLE_BOUNDS.items():
  if issue>=a+pd.Timedelta(hours=24) and issue+pd.Timedelta(hours=24)<b: return r
 return None

def build_clean(proton,xrs):
 rows=[]
 for issue in pd.date_range('2011-01-02','2017-11-29',freq='D',tz='UTC'):
  role=role_for(issue)
  if role is None: continue
  lab=label_issue(proton,issue)
  if not lab['resolved'] or lab['active']: continue
  pf=family_features(proton,issue,0,'proton'); xf=family_features(xrs,issue,0,'xrs')
  eligible=pf['p_coverage']>=.95 and pf['p_age']<=10 and xf['a_coverage']>=.95 and xf['b_coverage']>=.95 and xf['a_age']<=2 and xf['b_age']<=2
  if not eligible: continue
  rows.append({'issue':issue,'role':role,'y':lab['label'],'crossing_time':lab['crossing_time'],**pf,**xf})
 return pd.DataFrame(rows)

def model_cols(df):
 p=[c for c in df if c.startswith('p_')]; x=[c for c in df if c.startswith('a_') or c.startswith('b_')]; return p,x,p+x

def xgb_params(seed): return dict(n_estimators=300,learning_rate=.03,max_depth=3,min_child_weight=5,subsample=.8,colsample_bytree=.8,reg_lambda=1.,reg_alpha=0.,tree_method='hist',eval_metric='logloss',random_state=seed,n_jobs=2)
def fit_ensemble(X,y):
 ms=[]
 for s in SEEDS:
  m=XGBClassifier(**xgb_params(s)); m.fit(X,y); ms.append(m)
 return ms
def pred_ens(ms,X): return np.median(np.vstack([m.predict_proba(X)[:,1] for m in ms]),axis=0)
def logit(p): return np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-50,50)))
def fit_intercept(p,y):
 z=logit(p); f=lambda a: -np.sum(y*np.log(np.clip(sigmoid(z+a),1e-12,1))+(1-y)*np.log(np.clip(1-sigmoid(z+a),1e-12,1)))
 return float(minimize_scalar(f,bounds=(-10,10),method='bounded').x)
def calibrate(p,a): return sigmoid(logit(p)+a)
def counts(y,p,t):
 h=p>=t; return {'tp':int(np.sum((y==1)&h)),'fp':int(np.sum((y==0)&h)),'fn':int(np.sum((y==1)&~h)),'tn':int(np.sum((y==0)&~h))}
def tss(c):
 pod=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else np.nan; fpr=c['fp']/(c['fp']+c['tn']) if c['fp']+c['tn'] else np.nan; return pod-fpr
def far(c): return c['fp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0.
def best_tss(y,p):
 vals=np.unique(p); best=(-9,None,None)
 for th in vals:
  c=counts(y,p,th); z=tss(c)
  if z>best[0]+1e-12 or (abs(z-best[0])<=1e-12 and (best[1] is None or th>best[1])): best=(z,float(th),c)
 return best

def metric(y,p,th):
 c=counts(y,p,th); return {**c,'TSS':float(tss(c)),'FAR':float(far(c)),'Brier':float(brier_score_loss(y,p)),'AUPRC':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan}

def train(clean):
 pcols,xcols,jcols=model_cols(clean); roles={r:clean[clean.role==r].copy() for r in ['fit','calibration','threshold','score']}
 models={}; info={}; probs={}
 for name,cols in [('joint',jcols),('xrs',xcols),('proton',pcols)]:
  ms=fit_ensemble(roles['fit'][cols],roles['fit'].y.to_numpy()); models[name]=(ms,cols)
  rawcal=pred_ens(ms,roles['calibration'][cols]); a=fit_intercept(rawcal,roles['calibration'].y.to_numpy())
  probs[name]={};
  for r in roles: probs[name][r]=calibrate(pred_ens(ms,roles[r][cols]),a)
  z,th,c=best_tss(roles['threshold'].y.to_numpy(),probs[name]['threshold']); info[name]={'intercept':a,'threshold':th,'max_tss':z}
 fitprev=float(roles['fit'].y.mean()); prev2016=float(roles['threshold'].y.mean()); clim=np.full(len(roles['threshold']),fitprev); cb=float(brier_score_loss(roles['threshold'].y,clim))
 results=[]; gate=False
 for name in ['joint','xrs','proton']:
  y=roles['threshold'].y.to_numpy(); p=probs[name]['threshold']; met=metric(y,p,info[name]['threshold']); orig=met['AUPRC']>fitprev and met['Brier']<cb and met['TSS']>0; corr=met['AUPRC']>prev2016
  gate|=orig; results.append({'model':name,**met,'threshold':info[name]['threshold'],'fit_prevalence':fitprev,'prevalence_2016':prev2016,'AUPRC_gt_fit_prevalence_original':orig if (met['Brier']<cb and met['TSS']>0) else False,'AUPRC_gt_2016_prevalence':corr,'Brier_climatology_fit_constant':cb,'original_gate_pass':orig})
 # robustness logistic, not gate-controlling
 pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,solver='lbfgs',random_state=42))
 pipe.fit(roles['fit'][jcols],roles['fit'].y); lp=pipe.predict_proba(roles['threshold'][jcols])[:,1]; z,th,_=best_tss(roles['threshold'].y.to_numpy(),lp); results.append({'model':'logistic_robustness',**metric(roles['threshold'].y.to_numpy(),lp,th),'threshold':th,'fit_prevalence':fitprev,'prevalence_2016':prev2016,'Brier_climatology_fit_constant':cb,'original_gate_pass':False})
 return models,info,roles,probs,results,gate

def delayed_probs(models,info,base,proton,xrs,fam,delay):
 out={k:[] for k in ['joint','reduced','delayed_single','clean_joint']}; states=[]; ages=[]
 for _,row in base.iterrows():
  issue=row.issue; pf=family_features(proton,issue,delay if fam=='proton' else 0,'proton'); xf=family_features(xrs,issue,delay if fam=='xrs' else 0,'xrs')
  feature={**pf,**xf}
  for name in ['joint','xrs','proton']:
   ms,cols=models[name]; X=pd.DataFrame([{c:feature.get(c,np.nan) for c in cols}]); pr=float(calibrate(pred_ens(ms,X),info[name]['intercept'])[0])
   if name=='joint': out['joint'].append(pr)
   if fam=='xrs' and name=='proton': out['reduced'].append(pr)
   if fam=='xrs' and name=='xrs': out['delayed_single'].append(pr)
   if fam=='proton' and name=='xrs': out['reduced'].append(pr)
   if fam=='proton' and name=='proton': out['delayed_single'].append(pr)
  cj=float(calibrate(pred_ens(models['joint'][0],pd.DataFrame([{c:row[c] for c in models['joint'][1]}])),info['joint']['intercept'])[0]); out['clean_joint'].append(cj)
  ages.append(max(xf['a_age'],xf['b_age']) if fam=='xrs' else pf['p_age']); states.append(xf['b_last'] if fam=='xrs' else pf['p_last'])
 return {k:np.asarray(v) for k,v in out.items()},np.asarray(ages),np.asarray(states)

def threshold_for(name,fam,info):
 if name in ('joint','clean_joint'): return info['joint']['threshold']
 if name=='reduced': return info['proton' if fam=='xrs' else 'xrs']['threshold']
 return info['xrs' if fam=='xrs' else 'proton']['threshold']

def delay_table(models,info,roles,proton,xrs):
 rows=[]; store={}
 for role in ['threshold','score']:
  base=roles[role]; y=base.y.to_numpy()
  for fam in ['xrs','proton']:
   for d in DELAYS:
    ps,ages,states=delayed_probs(models,info,base,proton,xrs,fam,d); store[(role,fam,d)]=(ps,ages,states)
    for method,p in ps.items(): rows.append({'role':role,'delayed_family':fam,'delay_minutes':d,'method':method,**metric(y,p,threshold_for(method,fam,info))})
 return pd.DataFrame(rows),store

def bootstrap_delay(roles,store,info,reps=2000):
 rng=np.random.default_rng(20260909); out=[]; base=roles['score'].reset_index(drop=True); y=base.y.to_numpy(); episode=base.crossing_time.astype(str).to_numpy(); quiet=(base.issue.dt.floor('D').astype('int64')//int(7*86400*1e9)).to_numpy()
 pos_groups=[np.where((y==1)&(episode==e))[0] for e in np.unique(episode[y==1])]; neg_groups=[np.where((y==0)&(quiet==q))[0] for q in np.unique(quiet[y==0])]
 for fam in ['xrs','proton']:
  for d in DELAYS:
   ps=store[('score',fam,d)][0]
   for method in ['joint','reduced','delayed_single','clean_joint']:
    vals=[]; th=threshold_for(method,fam,info)
    for _ in range(reps):
     ix=np.concatenate([pos_groups[i] for i in rng.integers(0,len(pos_groups),len(pos_groups))]+[neg_groups[i] for i in rng.integers(0,len(neg_groups),len(neg_groups))])
     vals.append(tss(counts(y[ix],ps[method][ix],th)))
    out.append({'delayed_family':fam,'delay_minutes':d,'method':method,'TSS_ci_low':float(np.nanpercentile(vals,2.5)),'TSS_ci_high':float(np.nanpercentile(vals,97.5))})
   dif=[]; tj=info['joint']['threshold']; tr=info['proton' if fam=='xrs' else 'xrs']['threshold']
   for _ in range(reps):
    ix=np.concatenate([pos_groups[i] for i in rng.integers(0,len(pos_groups),len(pos_groups))]+[neg_groups[i] for i in rng.integers(0,len(neg_groups),len(neg_groups))])
    dif.append(tss(counts(y[ix],ps['reduced'][ix],tr))-tss(counts(y[ix],ps['joint'][ix],tj)))
   out.append({'delayed_family':fam,'delay_minutes':d,'method':'reduced_minus_joint','TSS_ci_low':float(np.nanpercentile(dif,2.5)),'TSS_ci_high':float(np.nanpercentile(dif,97.5))})
 return pd.DataFrame(out)

def action_metrics(y,alert,abstain):
 covered=~abstain; tp=int(np.sum((y==1)&alert&covered)); fp=int(np.sum((y==0)&alert&covered)); fn=int(np.sum((y==1)&((~alert)|abstain))); tn=int(np.sum((y==0)&(~alert)&covered)); unw=int(np.sum((y==1)&abstain))
 c={'tp':tp,'fp':fp,'fn':fn,'tn':tn}; return {**c,'TSS':float(tss(c)),'FAR':float(far(c)),'coverage':float(np.mean(covered)),'unwarned_positive_abstentions':unw}

def choose_reference(y,jp,rp,tj,tr):
 opts=[]
 for n,p,t in [('always_joint_stale',jp,tj),('always_reduced_fresh',rp,tr)]: opts.append((n,action_metrics(y,p>=t,np.zeros(len(y),bool))))
 minfn=min(x[1]['fn'] for x in opts); cand=[x for x in opts if x[1]['fn']==minfn]; cand.sort(key=lambda x:(x[1]['fp'],-x[1]['TSS'],0 if x[0]=='always_reduced_fresh' else 1)); return cand[0][0],opts

def policy_eval(roles,store,info):
 selection={}; score_rows=[]
 for fam in ['xrs','proton']:
  b16=roles['threshold']; y16=b16.y.to_numpy(); tj=info['joint']['threshold']; tr=info['proton' if fam=='xrs' else 'xrs']['threshold']
  J=np.concatenate([store[('threshold',fam,d)][0]['joint'] for d in NONZERO]); R=np.concatenate([store[('threshold',fam,d)][0]['reduced'] for d in NONZERO]); AGE=np.concatenate([store[('threshold',fam,d)][1] for d in NONZERO]); STATE=np.concatenate([store[('threshold',fam,d)][2] for d in NONZERO]); Y=np.tile(y16,len(NONZERO))
  ref,opts=choose_reference(Y,J,R,tj,tr); refp=J if ref=='always_joint_stale' else R; reft=tj if ref=='always_joint_stale' else tr; refmet=action_metrics(Y,refp>=reft,np.zeros(len(Y),bool))
  bestttl=None
  for ttl in DELAYS:
   usej=AGE<=ttl; a=np.where(usej,J>=tj,R>=tr); m=action_metrics(Y,a,np.zeros(len(Y),bool))
   if m['fn']<=refmet['fn'] and (bestttl is None or (m['fp'],ttl)<(bestttl[1]['fp'],bestttl[0])): bestttl=(ttl,m)
  bestmargin=None
  for mar in [0.,.01,.02,.05,.1]:
   abst=np.abs(refp-reft)<=mar; m=action_metrics(Y,refp>=reft,abst)
   if m['fn']<=refmet['fn'] and m['coverage']>=.8 and (bestmargin is None or (m['fp'],mar)<(bestmargin[1]['fp'],bestmargin[0])): bestmargin=(mar,m)
  bins=np.array([0,1,5,8,np.inf]) if fam=='proton' else np.array([0,1e-6,1e-5,1e-4,np.inf]); bttls=[]
  for bi in range(4):
   mask=np.isfinite(STATE)&(STATE>=bins[bi])&(STATE<bins[bi+1]); rmb=action_metrics(Y[mask],(refp[mask]>=reft),np.zeros(np.sum(mask),bool)) if np.any(mask) else {'fn':0,'fp':0}
   best=None
   for ttl in DELAYS:
    if not np.any(mask): continue
    al=np.where(AGE[mask]<=ttl,J[mask]>=tj,R[mask]>=tr); m=action_metrics(Y[mask],al,np.zeros(np.sum(mask),bool))
    if m['fn']<=rmb['fn'] and (best is None or (m['fp'],ttl)<(best[1]['fp'],best[0])): best=(ttl,m)
   bttls.append(None if best is None else best[0])
  selection[fam]={'reference':ref,'reference_metrics_2016_pooled':refmet,'fixed_ttl':None if bestttl is None else bestttl[0],'confidence_margin':None if bestmargin is None else bestmargin[0],'state_bins':bins.tolist(),'state_ttls':bttls,'reference_candidates_2016':opts}
  b17=roles['score']; y17=b17.y.to_numpy();
  for d in NONZERO:
   ps,age,state=store[('score',fam,d)]; jp,rp=ps['joint'],ps['reduced']; refp=jp if ref=='always_joint_stale' else rp; reft=tj if ref=='always_joint_stale' else tr
   actions={'reference':(refp>=reft,np.zeros(len(y17),bool))}
   if bestttl: actions['fixed_ttl']=(np.where(age<=bestttl[0],jp>=tj,rp>=tr),np.zeros(len(y17),bool))
   if bestmargin: actions['confidence_rejection']=(refp>=reft,np.abs(refp-reft)<=bestmargin[0])
   al=refp>=reft
   for bi,ttl in enumerate(bttls):
    if ttl is None: continue
    mask=np.isfinite(state)&(state>=bins[bi])&(state<bins[bi+1]); al[mask]=np.where(age[mask]<=ttl,jp[mask]>=tj,rp[mask]>=tr)
   actions['state_dependent_ttl']=(al,np.zeros(len(y17),bool))
   for n,(a,ab) in actions.items(): score_rows.append({'delayed_family':fam,'delay_minutes':d,'policy':n,**action_metrics(y17,a,ab)})
  # pooled 2017
  JP=np.concatenate([store[('score',fam,d)][0]['joint'] for d in NONZERO]); RP=np.concatenate([store[('score',fam,d)][0]['reduced'] for d in NONZERO]); AG=np.concatenate([store[('score',fam,d)][1] for d in NONZERO]); ST=np.concatenate([store[('score',fam,d)][2] for d in NONZERO]); YY=np.tile(y17,len(NONZERO)); RFP=JP if ref=='always_joint_stale' else RP; RFT=tj if ref=='always_joint_stale' else tr
  pol={'reference':(RFP>=RFT,np.zeros(len(YY),bool))}
  if bestttl: pol['fixed_ttl']=(np.where(AG<=bestttl[0],JP>=tj,RP>=tr),np.zeros(len(YY),bool))
  if bestmargin: pol['confidence_rejection']=(RFP>=RFT,np.abs(RFP-RFT)<=bestmargin[0])
  aa=RFP>=RFT
  for bi,ttl in enumerate(bttls):
   if ttl is None: continue
   mask=np.isfinite(ST)&(ST>=bins[bi])&(ST<bins[bi+1]); aa[mask]=np.where(AG[mask]<=ttl,JP[mask]>=tj,RP[mask]>=tr)
  pol['state_dependent_ttl']=(aa,np.zeros(len(YY),bool))
  refm=action_metrics(YY,*pol['reference'])
  for n,(a,ab) in pol.items():
   m=action_metrics(YY,a,ab); reduction=(refm['fp']-m['fp'])/refm['fp'] if refm['fp'] else 0.; m['false_alert_reduction_vs_reference']=float(reduction); m['additional_misses_vs_reference']=m['fn']-refm['fn']; m['practical_gate_pass']=bool(reduction>=.20 and m['additional_misses_vs_reference']==0 and m['coverage']>=.80); score_rows.append({'delayed_family':fam,'delay_minutes':'POOLED','policy':n,**m})
 return selection,pd.DataFrame(score_rows)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
 contracts={p.name:sha(p.read_bytes()) for p in [PRE,IMP,POOL,CLAR]}; dump(out/'contract_hashes.json',contracts)
 proton,xrs=acquire(out); clean=build_clean(proton,xrs); clean.to_csv(out/'clean_cohort.csv',index=False)
 counts_role=[]
 for r in ['fit','calibration','threshold','score']:
  z=clean[clean.role==r]; counts_role.append({'role':r,'issues':len(z),'positive_opportunities':int(z.y.sum()),'positive_episodes':int(z.loc[z.y==1,'crossing_time'].nunique()),'prevalence':float(z.y.mean()) if len(z) else np.nan})
 dump(out/'cohort_counts.json',counts_role)
 models,info,roles,clean_probs,clean_results,gate=train(clean); pd.DataFrame(clean_results).to_csv(out/'clean_model_results.csv',index=False); dump(out/'model_freeze_runtime.json',info)
 summary={'study_id':'IRIS_SEP_FRESHNESS_CROSSOVER_STUDY_V1','claim_boundary':'retrospective development evidence only; not an independent final test or operational replay','two_satellite_definition':'GOES-13 OMNI Av >10 MeV proton + GOES-15 NCEI operational XRS','cohort_counts':counts_role,'original_clean_signal_gate_pass':bool(gate),'clean_gate_interpretation':'screening authorization only; MAX_TSS>0 on 2016 is not independent evidence'}
 if not gate:
  summary['decision']='STOP_NO_CLEAN_SIGNAL'; dump(out/'summary.json',summary); print(json.dumps(summary,indent=2)); return
 dt,store=delay_table(models,info,roles,proton,xrs); dt.to_csv(out/'delay_performance.csv',index=False); boot=bootstrap_delay(roles,store,info); boot.to_csv(out/'delay_uncertainty.csv',index=False)
 sel,pol=policy_eval(roles,store,info); dump(out/'policy_selection_2016.json',sel); pol.to_csv(out/'policy_results_2017.csv',index=False)
 # crossover brackets from score TSS
 brackets={}
 for fam in ['xrs','proton']:
  q=dt[(dt.role=='score')&(dt.delayed_family==fam)]; j=q[q.method=='joint'].set_index('delay_minutes').TSS; rr=q[q.method=='reduced'].set_index('delay_minutes').TSS; dif=(rr-j).sort_index(); br=[]
  ds=list(dif.index)
  for a,b in zip(ds[:-1],ds[1:]):
   if dif.loc[a]<=0 and dif.loc[b]>0: br.append([int(a),int(b)])
  brackets[fam]={'differences':{str(int(k)):float(v) for k,v in dif.items()},'crossover_brackets':br}
 summary.update({'decision':'COMPLETED_FROZEN_DELAY_AND_POLICY_STUDY','crossover':brackets,'policy_selection':sel,'any_practical_gate_pass':bool(pol.get('practical_gate_pass',pd.Series(dtype=bool)).fillna(False).any())}); dump(out/'summary.json',summary); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
