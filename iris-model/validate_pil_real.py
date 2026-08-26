from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from astropy.io import fits

from data import build_records, cache_records
from preprocess import preprocess_fits


def hard_pil(b: np.ndarray, pixel_mm: float = 2.0, strong_g: float = 150.0):
    """Hard opposite-polarity contact and gradient diagnostics on fixed physical grid."""
    b=np.nan_to_num(b.astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    pos=b>=strong_g; neg=b<=-strong_g
    # One-pixel (2 Mm) contact neighborhood. A PIL pixel is strong positive/negative
    # interface contact, not merely a large derivative in a unipolar field.
    dp=binary_dilation(pos,iterations=1); dn=binary_dilation(neg,iterations=1)
    contact=dp & dn
    gy,gx=np.gradient(b,pixel_mm,pixel_mm)
    grad=np.sqrt(gx*gx+gy*gy)
    vals=grad[contact]
    return contact,grad,{
        'pil_contact_pixels':int(contact.sum()),
        'pil_length_proxy_mm':float(contact.sum()*pixel_mm),
        'pil_gradient_mean_g_per_mm':float(vals.mean()) if vals.size else 0.0,
        'pil_gradient_p90_g_per_mm':float(np.percentile(vals,90)) if vals.size else 0.0,
        'pil_gradient_max_g_per_mm':float(vals.max()) if vals.size else 0.0,
    }


def choose_group_audit(df: pd.DataFrame, n_pos_groups=10, n_neg_groups=10, seed=2026):
    """Deterministic TRAIN-only, group-stratified sample; one central-CMD row/group."""
    x=df[df.partition.eq('train')].copy()
    g=x.groupby('region_group_id').agg(group_positive=('label_m1plus_24h','max'),n=('sample_id','size')).reset_index()
    pos=g[g.group_positive.eq(1)].sort_values('region_group_id')
    neg=g[g.group_positive.eq(0)].sort_values('region_group_id')
    pos=pos.sample(n=min(n_pos_groups,len(pos)),random_state=seed) if len(pos) else pos
    neg=neg.sample(n=min(n_neg_groups,len(neg)),random_state=seed+1) if len(neg) else neg
    gids=set(pd.concat([pos,neg]).region_group_id)
    out=[]
    for gid in sorted(gids):
        z=x[x.region_group_id.eq(gid)].copy()
        # Prefer a positive row for a positive group, then choose closest-to-disk-center.
        if z.label_m1plus_24h.max()==1: z=z[z.label_m1plus_24h.eq(1)]
        z=z.assign(abs_cmd=z.cmd_deg.abs()).sort_values(['abs_cmd','t_rec','sample_id'])
        out.append(z.iloc[0])
    return pd.DataFrame(out).reset_index(drop=True)


def plot_overlay(b,contact,meta,outpath):
    fig,ax=plt.subplots(figsize=(6,5))
    lim=max(300,float(np.nanpercentile(np.abs(b),99)))
    im=ax.imshow(b,origin='lower',cmap='gray',vmin=-lim,vmax=lim)
    yy,xx=np.where(contact)
    if len(xx): ax.scatter(xx,yy,s=3,marker='s',facecolors='none',edgecolors='tab:red',linewidths=0.4)
    ax.set_title(f"{meta['sample_id']} | y={int(meta['label_m1plus_24h'])} | {meta['region_group_id']}")
    ax.set_xlabel('x [2 Mm/pixel]'); ax.set_ylabel('y [2 Mm/pixel]')
    fig.colorbar(im,ax=ax,label='LOS field [G]')
    fig.tight_layout(); fig.savefig(outpath,dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',default='cache/pil_audit'); ap.add_argument('--out-dir',default='pil_validation')
    args=ap.parse_args(); out=Path(args.out_dir); overlays=out/'overlays'; overlays.mkdir(parents=True,exist_ok=True)
    full=build_records(args.evidence_dir,'train')
    # build_records returns only train rows, but keep partition for helper compatibility
    full['partition']='train'
    audit=choose_group_audit(full,10,10,2026)
    audit=cache_records(audit,Path(args.cache_dir),workers=10)
    rows=[]
    for _,r in audit.iterrows():
        _,raw=preprocess_fits(r.fits_path,float(r.CDELT1),float(r.CDELT2),float(r.RSUN_REF))
        b=raw[0].numpy(); contact,grad,s=hard_pil(b,2.0,150.0)
        rec={'sample_id':r.sample_id,'region_group_id':r.region_group_id,'harpnum':int(r.harpnum),'t_rec':r.t_rec,'label_m1plus_24h':int(r.label_m1plus_24h),'cmd_deg':float(r.cmd_deg),'max_goes_class':str(r.max_goes_class),**s}
        rows.append(rec); plot_overlay(b,contact,rec,overlays/f"{r.sample_id}.png")
    result=pd.DataFrame(rows); result.to_csv(out/'real_pil_audit_samples.csv',index=False)
    summary={
      'scope':'TRAIN ONLY; deterministic one-row-per-connected-region audit',
      'n_samples':len(result),'n_positive':int(result.label_m1plus_24h.sum()),'n_negative':int((1-result.label_m1plus_24h).sum()),
      'positive_mean_gradient_g_per_mm':float(result[result.label_m1plus_24h.eq(1)].pil_gradient_mean_g_per_mm.mean()),
      'negative_mean_gradient_g_per_mm':float(result[result.label_m1plus_24h.eq(0)].pil_gradient_mean_g_per_mm.mean()),
      'all_finite':bool(np.isfinite(result.filter(like='gradient').to_numpy()).all()),
      'contact_nonzero_samples':int((result.pil_contact_pixels>0).sum()),
      'interpretation':'Engineering validation of the PIL implementation on real definitive SHARP fields. Not a downstream flare-performance result and not a replacement for student visual audit.'
    }
    (out/'real_pil_validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
