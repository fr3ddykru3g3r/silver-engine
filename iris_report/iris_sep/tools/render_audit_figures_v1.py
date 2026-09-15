"""Render disclosed technical graphics from audit aggregate outputs only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'audit_20260915'
COLORS=['#0072B2','#D55E00','#009E73']
MODELS=['xgb_joint','xgb_no_proton','past_proton_ge10_proxy']
NAMES=['Joint XGBoost','No-proton XGBoost','Past-proton proxy']
FOOT='Historical, development-exposed diagnostic · 85 episodes · complete horizons before 10 Sep 2025\nAI-assisted graphic: deterministic Matplotlib code; frozen predictions; no model refitting'


def save(fig,name):
    fig.text(.06,.02,FOOT,fontsize=8,color='#444444',va='bottom')
    fig.subplots_adjust(bottom=.22,top=.84,left=.12,right=.95)
    for ext in ['png','pdf','svg']:
        fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    r=json.loads((ROOT/'results/strict_preboundary_results.json').read_text())
    f=json.loads((ROOT/'results/followups_results.json').read_text())
    table=pd.read_csv(ROOT/'results/strict_preboundary_metrics.csv')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(10,6))
    views=['MAPPED_STANDARD','EPISODE_NORMALIZED_OCCURRENCE','NEW_ONSET']
    for model,name,c,marker in zip(MODELS,NAMES,COLORS,['o','s','^']):
        vals=table[table.model.eq(model)].set_index('view').reindex(views).tss.to_numpy()
        ax.plot(range(3),vals,marker=marker,color=c,label=name,lw=2.2,ms=8)
        for x,y in enumerate(vals):
            offset = -18 if model == 'past_proton_ge10_proxy' and x == 1 else 10
            if model == 'xgb_joint' and x == 2: offset = -18
            if model == 'xgb_no_proton' and x == 0: offset = -18
            ax.annotate(f'{y:.3f}',(x,y),xytext=(0,offset),textcoords='offset points',ha='center',color=c)
    ax.set_xticks(range(3),['Mapped\noccurrence','Episode-normalized\noccurrence','New onset'])
    ax.set_ylim(-.02,.90);ax.set_ylabel('TSS = sensitivity − false-positive rate')
    ax.axhline(0,color='#888888',lw=.7);ax.legend(loc='upper right',frameon=False)
    fig.suptitle('Fixed predictions, different questions',x=.12,ha='left',fontsize=19,fontweight='bold')
    ax.grid(axis='y',alpha=.15);save(fig,'01_fixed_predictions')
    fig,ax=plt.subplots(figsize=(10,6))
    y=np.arange(3);g=np.array([r['mechanisms'][m]['multiplicity_gap'] for m in MODELS]);h=np.array([r['mechanisms'][m]['persistence_term'] for m in MODELS])
    # Grouped bars show negative contributions without ambiguous stacked origins.
    ax.barh(y+.16,g,height=.28,color=COLORS[0],label='Multiplicity term')
    ax.barh(y-.16,h,height=.28,color=COLORS[1],hatch='//',label='Persistence term')
    for i in range(3):
        ax.text(max(g[i],0)+.006,y[i]+.16,f'{g[i]:+.3f}',va='center',fontsize=10)
        ax.text(h[i]+.006,y[i]-.16,f'{h[i]:+.3f}',va='center',fontsize=10)
    ax.set_yticks(y,NAMES);ax.invert_yaxis();ax.axvline(0,color='#777777',lw=.8)
    ax.set_xlim(-.03,.43);ax.set_xlabel('Contribution to mapped TSS − onset TSS')
    ax.legend(loc='lower right',frameon=False);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Exact attribution on the matched cohort',x=.12,ha='left',fontsize=18,fontweight='bold')
    save(fig,'02_decomposition')
    fig,ax=plt.subplots(figsize=(10,6))
    labels=[];intervals=[];colors=[]
    for g in ['27','90','quarter','year']:
        for name,c,key in [('Multiplicity',COLORS[0],'normalized_minus_mapped'),('Persistence',COLORS[1],'onset_minus_normalized')]:
            labels.append(f'{g} — {name}');intervals.append(f['temporal_bootstrap'][g]['results']['xgb_joint'][key]);colors.append(c)
    for i,(z,c) in enumerate(zip(intervals,colors)):
        lo,hi=z['interval95'];ax.plot([lo,hi],[i,i],color=c,lw=2.5);ax.plot(z['median'],i,'o',color=c)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,color='#666666',linestyle='--')
    ax.set_xlabel('TSS contrast (later view − earlier view): median and 95% interval')
    ax.set_xlim(-.30,.025);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Signs persist when nearby events are resampled together',x=.12,ha='left',fontsize=16,fontweight='bold')
    save(fig,'03_temporal_intervals')

if __name__=='__main__':main()
