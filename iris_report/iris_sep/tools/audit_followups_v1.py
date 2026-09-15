"""Bounded post-hoc robustness and archive checks. Never trains or reads feeds."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from tools.audit_estimands_v1 import safe_rows, pinned_zip, validate, REPLAY_SHA, MEMBER, interval, metrics

ARCHIVES={
    '10275033377':'4fce9da1be1764b2cab4a59e6398bdf6642b8dbe6f7d5cd7246d8b24128e9984',
    '10277940827':'6acce6e7916f8aaaf735fd786d59b3d5283391b734d0bab0f438d14ffe25701f',
    '10277871843':'c7756a24fee8519b5975bd6a0cac8e3f83303c88389e4a476ae500e7ea7f96ad',
    '10311704457':'eb74943a320d63d7e57b48af9a69d333dfc7cbaa38d87518c735d4dcd8080996',
}


def temporal_contrasts(frame, grouping, draws=10000, seed=20260916):
    base=frame[frame.model=='xgb_joint']
    onset=base[base.eligibility=='ELIGIBLE_ONSET_POSITIVE'].set_index('episode_id')
    ids=sorted(onset.index)
    times=pd.to_datetime(onset.reindex(ids).issue_timestamp,utc=True)
    # Full calendar bins, including empty bins, prevent conditioning on activity.
    lo,hi=base.issue_timestamp.min(),base.issue_timestamp.max()
    if grouping=='year':
        keys=times.dt.year.to_numpy(); universe=np.arange(lo.year,hi.year+1)
    elif grouping=='quarter':
        keys=(times.dt.year*4+(times.dt.month-1)//3).to_numpy()
        universe=np.arange(lo.year*4+(lo.month-1)//3,hi.year*4+(hi.month-1)//3+1)
    else:
        days=int(grouping);origin=pd.Timestamp('2005-01-01',tz='UTC')
        keys=((times-origin).dt.days//days).to_numpy()
        universe=np.arange((lo-origin).days//days,(hi-origin).days//days+1)
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(universe),(draws,len(universe)))
    output={}
    for model in ['xgb_joint','past_proton_ge10_proxy','xgb_no_proton']:
        g=frame[(frame.model==model)&frame.standard_label.eq(1)]
        ep=g.groupby('episode_id').alert.agg(['size','sum','mean']).reindex(ids)
        u=g[g.eligibility=='ELIGIBLE_ONSET_POSITIVE'].set_index('episode_id').alert.reindex(ids)
        # N, hits, sum r_i, sum onset hits, number of episodes by time bin.
        vals=pd.DataFrame({'key':keys,'n':ep['size'].to_numpy(),'h':ep['sum'].to_numpy(),
                           'r':ep['mean'].to_numpy(),'u':u.to_numpy(),'k':np.ones(len(ids))})
        totals=vals.groupby('key')[['n','h','r','u','k']].sum().reindex(universe,fill_value=0).to_numpy(float)
        d=totals[samples].sum(axis=1); good=(d[:,0]>0)&(d[:,4]>0);d=d[good]
        row=d[:,1]/d[:,0];normalized=d[:,2]/d[:,4];on=d[:,3]/d[:,4]
        output[model]={'normalized_minus_mapped':interval(normalized-row),
                       'onset_minus_normalized':interval(on-normalized),
                       'onset_minus_mapped':interval(on-row),'valid_draws':len(d)}
    return {'grouping':str(grouping),'calendar_bins':len(universe),'positive_episodes':len(ids),
            'draws':draws,'seed':seed,'conditional_on_fixed_forecasts':True,
            'stationarity_not_established':True,'results':output}


def complete_subregimes(months):
    """Do not discard a complete run because its adjacent month is incomplete."""
    runs=[]; current=None
    for m in sorted(months,key=lambda x:x['year_month']):
        year,month=map(int,m['year_month'].split('-')); ordinal=year*12+month
        eligible=(m.get('complete_daily_coverage') is True and len(m.get('prefixes',[]))==1
                  and len(m.get('versions',[]))==1)
        key='|'.join(m['prefixes']+m['versions']) if eligible else None
        if not eligible:
            current=None; continue
        if current is None or current['key']!=key or current['_last']!=ordinal-1:
            current={'key':key,'start_month':m['year_month'],'end_month':m['year_month'],'months':1,'_last':ordinal}
            runs.append(current)
        else:
            current['end_month']=m['year_month'];current['months']+=1;current['_last']=ordinal
    return [{k:v for k,v in r.items() if k!='_last'} for r in runs]


def independent_controller(times,y,p,floor,q):
    t=pd.to_datetime(times,utc=True); thresholds=[]
    for now in t:
        ix=np.flatnonzero(((t>=now-pd.Timedelta(days=120)) &
                           (t+pd.Timedelta(hours=25)<=now)).to_numpy() & (y==0))
        scores=np.sort(p[ix][np.isfinite(p[ix])])
        th=floor if len(scores)<30 else max(floor,float(scores[int(np.ceil(q*(len(scores)-1)))]))
        thresholds.append(th)
    return np.array(thresholds)


def controller_audit(archives):
    results={}; base_p=None; base_y=None
    for aid,member,q in [('10275033377','phase2_score_predictions.csv',None),
                          ('10277940827','fpr_controller_replay.csv',.85),
                          ('10277871843','fpr_controller_v2_replay.csv',.80)]:
        with pinned_zip(archives/(aid+'.zip'),ARCHIVES[aid]) as z:
            f=pd.DataFrame(safe_rows(io.TextIOWrapper(z.open(member)),reject_any=True))
            if q is None:
                summary=json.loads(z.read('phase2_summary.json'))
                results['phase2_all_models']={}
                base_y=f.y_onset.to_numpy(int);base_p=f['p__direct_onset_engineered'].to_numpy(float)
                for model,th in summary['threshold_selection'].items():
                    prob=f['p__'+model].to_numpy(float);threshold=th['chosen']['threshold']
                    met=metrics(base_y,prob>=threshold,prob)
                    results['phase2_all_models'][model]={'threshold':threshold,**met}
                continue
        y=f.y_onset.to_numpy(int);p=f.probability.to_numpy(float)
        if not np.array_equal(y,base_y) or not np.allclose(p,base_p,rtol=0,atol=1e-15):
            raise ValueError('controller artifact did not preserve Phase-II probabilities')
        th=independent_controller(f.issue_timestamp,y,p,float(f.base_threshold.iloc[0]),q)
        if not np.allclose(th,f.controller_threshold.to_numpy(float),rtol=0,atol=1e-15):
            raise ValueError('causal threshold reconstruction mismatch')
        alert=(p>=th).astype(int)
        if not np.array_equal(alert,f.controller_alert.to_numpy(int)):
            raise ValueError('controller alert reconstruction mismatch')
        results[aid]={'quantile':q,'metrics':metrics(y,alert,p),
                       'probabilities_match_frozen_phase2':True,'thresholds_independently_reconstructed':True,
                       'historical_label_availability_assumed_25h_not_operationally_verified':True}
    return results


def run(archives,out):
    out.mkdir(parents=True,exist_ok=False)
    with pinned_zip(archives/'10137507101.zip',REPLAY_SHA) as z:
        f=validate(pd.DataFrame(safe_rows(io.TextIOWrapper(z.open(MEMBER)))))
    result={'status':'PASS','evidence_class':'POSTHOC_STRICT_PREBOUNDARY_DIAGNOSTIC',
            'temporal_bootstrap':{str(g):temporal_contrasts(f,g) for g in [27,90,'quarter','year']},
            'controller_audit':controller_audit(archives)}
    with pinned_zip(archives/'10311704457.zip',ARCHIVES['10311704457']) as z:
        original=json.loads(z.read('sgps_l2_source_regimes_v1.json'))
    runs=complete_subregimes(original['months']); eligible=[r for r in runs if r['months']>=12]
    result['source_inventory_correction']={'original_longest':original['longest_complete_homogeneous_regime'],
        'corrected_complete_runs':runs,'longest_complete':max(eligible,key=lambda x:(x['months'],x['end_month'])) if eligible else None,
        'source_only':True,'training_authorized':False,'live_equivalence_established':False,
        'reason':'Original scanner grouped incomplete endpoint months with complete months, then rejected the entire regime.'}
    result['exact_binomial_pod_interval_one_detection_one_event']=[.025,1.0]
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archives',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();r=run(a.archives,a.output);print(r['status'])
