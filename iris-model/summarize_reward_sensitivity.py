from __future__ import annotations

import argparse,json,re
from pathlib import Path
import pandas as pd
import numpy as np


def parse_name(name:str):
    if name.startswith('tp_'):return 'tp',float(name.split('_')[1])
    if name.startswith('tn_'):return 'tn',float(name.split('_')[1])
    if name.startswith('fp_m'):return 'fp',-float(name.split('m')[1])
    if name.startswith('fn_m'):return 'fn',-float(name.split('m')[1])
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
    for p in sorted(root.rglob('metrics.json')):
        name=p.parent.name
        if not (name.startswith('tp_') or name.startswith('tn_') or name.startswith('fp_m') or name.startswith('fn_m')):continue
        d=json.loads(p.read_text());dim,val=parse_name(name);m=d['validation_selected'];m05=d['validation_at_0.5'];cfg=d.get('paper_config',{})
        rows.append({'name':name,'dimension':dim,'value':val,'reward_tp':cfg.get('tp'),'reward_tn':cfg.get('tn'),'reward_fp':cfg.get('fp'),'reward_fn':cfg.get('fn'),'threshold':d['validation_selected_threshold'],'tss':m.get('tss'),'hss':m.get('hss'),'auroc':m.get('auroc'),'auprc':m.get('auprc'),'brier':m.get('brier'),'bss':m.get('bss'),'ece10':m.get('ece10'),'tss_at_0.5':m05.get('tss'),'auroc_at_0.5':m05.get('auroc'),'updates':sum(int(h.get('updates',0)) for h in d.get('history',[])),'episodes':d.get('episodes_run'),'train_items':d.get('train_items'),'validation_items':d.get('validation_items')})
    df=pd.DataFrame(rows).sort_values(['dimension','value']).reset_index(drop=True)
    if len(df)!=40:raise RuntimeError(f'expected 40 reward perturbations, found {len(df)}')
    df.to_csv(out/'reward_sensitivity_all.csv',index=False)
    base=df[(df.reward_tp==10)&(df.reward_tn==4)&(df.reward_fp==-20)&(df.reward_fn==-15)].copy()
    base_consistency={k:float(base[k].max()-base[k].min()) for k in ['tss','auroc','brier']}
    dims={}
    for dim,g in df.groupby('dimension'):
        bt=g.sort_values(['tss','auroc'],ascending=[False,False]).iloc[0];ba=g.sort_values(['auroc','tss'],ascending=[False,False]).iloc[0]
        dims[dim]={'n':len(g),'value_range':[float(g.value.min()),float(g.value.max())],'best_tss':{'value':float(bt.value),'tss':float(bt.tss),'auroc':float(bt.auroc),'threshold':float(bt.threshold)},'best_auroc':{'value':float(ba.value),'auroc':float(ba.auroc),'tss':float(ba.tss),'threshold':float(ba.threshold)},'tss_range':[float(g.tss.min()),float(g.tss.max())],'auroc_range':[float(g.auroc.min()),float(g.auroc.max())], 'spearman_value_tss':float(g[['value','tss']].corr(method='spearman').iloc[0,1]),'spearman_value_auroc':float(g[['value','auroc']].corr(method='spearman').iloc[0,1])}
    overall=df.sort_values(['tss','auroc'],ascending=[False,False]).head(10)[['name','reward_tp','reward_tn','reward_fp','reward_fn','threshold','tss','hss','auroc','brier','bss','ece10']].to_dict('records')
    rep={'experiment':'validation-only one-at-a-time CDR Transformer-10 reward sensitivity','rows':len(df),'baseline_reward':[10,4,-20,-15],'baseline_duplicate_runs':len(base),'baseline_consistency_max_minus_min':base_consistency,'dimensions':dims,'top10_by_tss':overall,'warning':'This grid used reduced-update-rate resource mode and validation only; it is sensitivity evidence, not a new test-selected model.'}
    (out/'reward_sensitivity_summary.json').write_text(json.dumps(rep,indent=2,allow_nan=True)+'\n')
    print(json.dumps(rep,indent=2,allow_nan=True))

if __name__=='__main__':main()
