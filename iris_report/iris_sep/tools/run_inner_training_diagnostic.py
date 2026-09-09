"""Bounded rolling legacy-target diagnostic using outer role=train exclusively."""
from __future__ import annotations
import argparse, hashlib, json, time
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from iris_report.iris_sep.tools.run_local_xgboost import EXPECTED_SOURCE_SHA256, EXPECTED_MANIFEST_SHA256, EXPECTED_FEATURE_MANIFEST_SHA256, META, TARGET
from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import apply_calibration,fit_intercept_calibration,select_tss_threshold,threshold_metrics,probability_metrics,minimum_far_at_pod
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[7,13,26,42,73]
ROLES=['fit','stop','calibration','threshold','score']

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_train():
    source=ROOT/'data_processed/sepnet_v1_development_v3.csv'
    manifest=ROOT/'receipts/sepnet_v1_development_v3_manifest.json'
    if digest(source)!=EXPECTED_SOURCE_SHA256 or digest(manifest)!=EXPECTED_MANIFEST_SHA256:raise ValueError('pinned source mismatch')
    # This is a preverified development-only file, never a mixed locked table.
    frame=pd.read_csv(source)
    frame=frame.loc[frame.role=='train'].sort_values('window_end').reset_index(drop=True)
    features=[c for c in frame if c not in META]
    if hashlib.sha256(json.dumps(features,separators=(',',':')).encode()).hexdigest()!=EXPECTED_FEATURE_MANIFEST_SHA256:raise ValueError('feature schema mismatch')
    return frame,features

def folds(frame):
    if set(frame.role)!={'train'} or frame.issue_id.duplicated().any():raise ValueError('outer role or identity violation')
    units=frame.groupby('unit_id').agg(start=('window_end','min'),end=('window_end','max')).sort_values('start')
    n=len(units);result=[]
    for fraction in [.55,.70,.85,1.0]:
        end=int(n*fraction);cuts=[0,int(end*.6),int(end*.7),int(end*.8),int(end*.9),end]
        role_indices={}; removed=[];previous=None
        for role,a,b in zip(ROLES,cuts,cuts[1:]):
            block=units.iloc[a:b]
            if previous is not None:
                keep=pd.to_datetime(block.start,utc=True)>previous+pd.Timedelta(hours=24)
                removed+=block.index[~keep].tolist();block=block[keep]
            if block.empty:raise ValueError('empty inner role')
            ix=np.flatnonzero(frame.unit_id.isin(block.index).to_numpy())
            if frame.iloc[ix][TARGET].nunique()!=2:raise ValueError('both classes required in every role')
            role_indices[role]=ix.tolist();previous=pd.to_datetime(block.end,utc=True).max()
        all_ix=[x for values in role_indices.values() for x in values]
        if len(all_ix)!=len(set(all_ix)):raise ValueError('inner role overlap')
        used=set(frame.iloc[all_ix].unit_id);expected=set(units.iloc[:end].index)
        if used.intersection(removed) or used.union(removed)!=expected:raise ValueError('coverage failure')
        result.append({'indices':role_indices,'purged_units':removed,'prefix_units':end})
    scores=[i for f in result for i in f['indices']['score']]
    if len(scores)!=len(set(scores)):raise ValueError('score folds overlap')
    return result

def prepare(out):
    if out.exists():raise ValueError('immutable run directory exists')
    frame,features=load_train(); split=folds(frame);out.mkdir(parents=True)
    config={'scope':'TRAIN_ONLY_LEGACY_TARGET_DIAGNOSTIC_NOT_FINAL','source_sha256':EXPECTED_SOURCE_SHA256,'seeds':SEEDS,
      'arms':['climatology','elastic_net','xgboost','compact','compact_signed_log'],
      'fold_prefix_fractions':[.55,.70,.85,1.0],'role_fractions':[.6,.1,.1,.1,.1],
      'purge_hours':24,'strict_purge':True,'max_neural_epochs':200,'patience':20,'batch_size':256,
      'neural_variant':'sign(x)*log1p(abs(x)); preserve NaN masks; train-fitted normalization',
      'new_hyperparameter_search':False,'selection':'Diagnostic only; no winner or outer-monitor rerun; retain every arm.',
      'outer_roles_accessed_for_metrics':['train'],'locked_test_accessed':False,'features':features,'runner_sha256':digest(__file__),'dependency_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'workstreams/luna_inner_neural_20260905/helper.py',ROOT/'tools/train_tabular_multibranch.py',ROOT/'src/iris_sep/modeling/tabular_multibranch.py',ROOT/'workstreams/luna_i_eval_ops/evaluation.py']}}
    save(out/'preregistration.json',config);save(out/'folds.json',split)
    save(out/'fold_support.json',[{r:{'rows':len(ix),'positives':int(frame.iloc[ix][TARGET].sum()),'units':int(frame.iloc[ix].unit_id.nunique()),'from':frame.iloc[ix].window_end.min(),'to':frame.iloc[ix].window_end.max()} for r,ix in f['indices'].items()} for f in split])
    return config

def execute(out):
    start=time.monotonic();config=json.loads((out/'preregistration.json').read_text())
    if config['runner_sha256']!=digest(__file__):raise ValueError('runner changed since freeze')
    if any(digest(ROOT/p)!=h for p,h in config['dependency_sha256'].items()):raise ValueError('dependency changed since freeze')
    if (out/'receipt.json').exists():raise ValueError('completed immutable run')
    frame,features=load_train();fs=json.loads((out/'folds.json').read_text())
    if fs!=folds(frame):raise ValueError('fold mutation')
    from iris_report.iris_sep.workstreams.luna_inner_neural_20260905.helper import fit_predict
    records=[];metrics=[];y=frame[TARGET].to_numpy()
    for fold_no,fold in enumerate(fs):
      ix={k:np.array(v) for k,v in fold['indices'].items()};train=frame.iloc[ix['fit']];prev=float(train[TARGET].mean())
      predix=np.concatenate([ix[r] for r in ['calibration','threshold','score']]);offset=np.cumsum([0]+[len(ix[r]) for r in ['calibration','threshold','score']]);slices={r:slice(offset[j],offset[j+1]) for j,r in enumerate(['calibration','threshold','score'])}
      for arm in config['arms']:
        folder=out/f'fold_{fold_no}'/arm;folder.mkdir(parents=True,exist_ok=True);all_p=[];seed_meta=[];arm_failure=None
        for seed in (SEEDS if arm not in ['climatology'] else [7]):
          seedpath=folder/f'seed_{seed}.npz';metapath=folder/f'seed_{seed}.json'
          if seedpath.exists() and metapath.exists():
            meta=json.loads(metapath.read_text());assert digest(seedpath)==meta['prediction_sha256'];raw=np.load(seedpath)['logits']
          else:
            if arm=='climatology':raw=np.full(len(predix),np.log(prev/(1-prev)));meta={}
            elif arm=='elastic_net':
              model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=1,max_iter=10000,random_state=seed,n_jobs=1))
              model.fit(train[features],train[TARGET]);raw=model.decision_function(frame.iloc[predix][features]);joblib.dump(model,folder/f'model_{seed}.joblib');np.testing.assert_array_equal(raw,joblib.load(folder/f'model_{seed}.joblib').decision_function(frame.iloc[predix][features]));meta={'iterations':model[-1].n_iter_.tolist()}
              if max(meta['iterations'])>=10000:raise ValueError('elastic net did not converge')
            elif arm=='xgboost':
              model=XGBClassifier(n_estimators=2000,learning_rate=.03,max_depth=3,min_child_weight=5,subsample=.8,colsample_bytree=.8,reg_lambda=1,reg_alpha=0,objective='binary:logistic',eval_metric='aucpr',tree_method='hist',early_stopping_rounds=50,scale_pos_weight=(1-prev)/prev,n_jobs=1,random_state=seed)
              model.fit(train[features],train[TARGET],eval_set=[(frame.iloc[ix['stop']][features],y[ix['stop']])],verbose=False)
              raw=model.predict(frame.iloc[predix][features],output_margin=True);meta={'best_iteration':int(model.best_iteration)};model.save_model(folder/f'model_{seed}.json')
            else:
              nf=frame.copy()
              if arm=='compact_signed_log':
                values=nf[features].to_numpy(dtype=float);nf[features]=np.sign(values)*np.log1p(np.abs(values))
              raw,meta=fit_predict(nf,features,ix['fit'],ix['stop'],predix,seed,folder/f'model_{seed}')
            np.savez(seedpath,logits=raw);meta['prediction_sha256']=digest(seedpath);save(metapath,meta)
          if not np.isfinite(raw).all():
            arm_failure={'fold':fold_no,'arm':arm,'status':'FAILED_NONFINITE_LOGITS','seed':seed,
                         'finite_logits':int(np.isfinite(raw).sum()),'total_logits':int(len(raw)),
                         'artifact_sha256':digest(seedpath)}
            save(folder/'failure.json',arm_failure)
            break
          cal=fit_intercept_calibration(raw[slices['calibration']],y[ix['calibration']],role='validation_calibration')
          all_p.append(apply_calibration(raw,cal));seed_meta.append({**meta,'seed':seed,'calibration_intercept':float(cal.intercept)})
        if arm_failure is not None:
          metrics.append(arm_failure);print(json.dumps(arm_failure),flush=True);continue
        p=np.median(all_p,axis=0);threshold=select_tss_threshold(y[ix['threshold']],p[slices['threshold']],role='validation_threshold').threshold
        scorep=p[slices['score']];scorey=y[ix['score']]
        result={'fold':fold_no,'arm':arm,'threshold':float(threshold),**threshold_metrics(scorey,scorep,threshold),**probability_metrics(scorey,scorep,reference_probability=prev),'matched_POD_0_8':minimum_far_at_pod(scorey,scorep,.8),'seeds':seed_meta}
        save(folder/'result.json',result);metrics.append(result)
        for idx,prob in zip(ix['score'],scorep):records.append({'fold':fold_no,'arm':arm,'issue_id':frame.iloc[idx].issue_id,'unit_id':frame.iloc[idx].unit_id,'label':int(y[idx]),'probability':float(prob),'threshold':float(threshold),'prediction':int(prob>=threshold),'outer_role':'train'})
        print(json.dumps({'fold':fold_no,'arm':arm,'TSS':result['TSS'],'FAR':result['FAR'],'BRIER':result['BRIER']}),flush=True)
    predpath=out/'inner_predictions.csv';pd.DataFrame(records).to_csv(predpath,index=False,float_format='%.17g')
    saved=pd.read_csv(predpath,float_precision='round_trip');summary={}
    for arm,g in saved.groupby('arm'):
      # Different preselected thresholds per fold; aggregate confusion counts, not threshold tuning.
      yy=g.label.to_numpy();pp=g.prediction.to_numpy();tp=int(((yy==1)&(pp==1)).sum());fp=int(((yy==0)&(pp==1)).sum());fn=int(((yy==1)&(pp==0)).sum());tn=int(((yy==0)&(pp==0)).sum())
      summary[arm]={'TP':tp,'FP':fp,'FN':fn,'TN':tn,'TSS':tp/(tp+fn)-fp/(fp+tn),'FAR':fp/(tp+fp) if tp+fp else None,'BRIER':float(np.mean((g.probability-g.label)**2)),'rows':len(g)}
    receipt={'status':'COMPLETED_WITH_RETAINED_ARM_FAILURES_TRAIN_ONLY_LEGACY_DIAGNOSTIC','locked_test_accessed':False,'outer_monitor_scored':False,'new_crossing_evidence':False,'elapsed_seconds':time.monotonic()-start,'summary':summary,'failures':[m for m in metrics if m.get('status','').startswith('FAILED_')],'fold_results':[m for m in metrics if not m.get('status','').startswith('FAILED_')],'prediction_sha256':digest(predpath),'preregistration_sha256':digest(out/'preregistration.json'),'folds_sha256':digest(out/'folds.json'),'superiority_established':False}
    save(out/'receipt.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepare(a.output) if a.mode=='prepare' else execute(a.output)
