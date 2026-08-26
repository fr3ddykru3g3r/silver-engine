from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import build_records, cache_records, MagnetogramDataset
from train_generator_v2 import group_temporal_subset
from preprocess import denormalize_gauss

EPS=1e-9


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    x=x.reshape(x.shape[0],-1).float(); y=y.reshape(y.shape[0],-1).float()
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    return float((2*torch.cdist(x,y).mean()-torch.cdist(x,x).mean()-torch.cdist(y,y).mean()).item())


def robust_standardize(real: np.ndarray, syn: np.ndarray):
    med=np.nanmedian(real,axis=0)
    q1=np.nanpercentile(real,25,axis=0); q3=np.nanpercentile(real,75,axis=0)
    scale=np.where((q3-q1)>1e-6,q3-q1,np.nanstd(real,axis=0)+1e-6)
    scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.0)
    return (real-med)/scale,(syn-med)/scale,med,scale


def split_half_baseline(real_standardized: np.ndarray, seed: int, repeats: int = 64) -> dict:
    """Calibrate a same-distribution distance scale using only real training data."""
    rng=np.random.default_rng(seed)
    vals=[]
    n=len(real_standardized)
    if n < 8:
        return {'median':float('nan'),'p90':float('nan'),'repeats':0}
    for _ in range(repeats):
        idx=rng.permutation(n)
        a=real_standardized[idx[:n//2]]; b=real_standardized[idx[n//2:]]
        vals.append(energy_distance(torch.from_numpy(a),torch.from_numpy(b)))
    x=np.asarray(vals,float)
    return {'median':float(np.nanmedian(x)),'p90':float(np.nanpercentile(x,90)),'repeats':len(x)}


def load_real(evidence_dir,cache_dir,per_group,seed):
    rec=build_records(evidence_dir,'train')
    rec=rec[rec.label_m1plus_24h.eq(1)].copy()
    rec=group_temporal_subset(rec,per_group,per_group,seed)
    rec=cache_records(rec,Path(cache_dir),workers=12)
    ds=MagnetogramDataset(rec)
    raw=[]; lat=[]
    for i in range(len(ds)):
        z=ds[i]; raw.append(z['raw_gauss']); lat.append(float(z['latitude']))
    return torch.stack(raw),np.asarray(lat,np.float32),rec


def load_synthetic(manifest):
    p=Path(manifest); m=pd.read_csv(p); arr=[]
    for raw in m.array_path:
        q=Path(str(raw))
        if not q.exists(): q=p.parent/'arrays'/q.name
        arr.append(np.load(q).astype(np.float32))
    x=torch.from_numpy(np.stack(arr))[:,None]
    return denormalize_gauss(x),m.latitude_deg.to_numpy(np.float32),m


def gradient(b,pixel_mm=2.0):
    bx=F.pad(b,(1,1,0,0),mode='replicate'); by=F.pad(b,(0,0,1,1),mode='replicate')
    gx=(bx[:,:,:,2:]-bx[:,:,:,:-2])/(2*pixel_mm)
    gy=(by[:,:,2:,:]-by[:,:,:-2,:])/(2*pixel_mm)
    return torch.sqrt(gx.square()+gy.square()+1e-12)


def hard_pil_descriptor(b,strong=150.0,radius_px=2,pixel_mm=2.0):
    pos=(b>=strong).float(); neg=(b<=-strong).float(); k=2*radius_px+1
    dp=F.max_pool2d(pos,k,1,radius_px); dn=F.max_pool2d(neg,k,1,radius_px)
    contact=(dp>0.5)&(dn>0.5); g=gradient(b,pixel_mm)
    rows=[]
    for i in range(len(b)):
        mask=contact[i,0]
        vals=g[i,0][mask]
        if len(vals)==0:
            rows.append([0.,0.,0.,0.,0.,0.,0.,0.]); continue
        q90=torch.quantile(vals,0.90); top=vals[vals>=q90]
        rows.append([
            float(vals.mean()),float(torch.sqrt((vals.square()).mean())),float(top.mean()),
            float((vals>100).float().mean()),float((vals>250).float().mean()),float((vals>500).float().mean()),
            float(mask.float().mean()),1.0,
        ])
    return np.asarray(rows,np.float32)


def hard_geometry_descriptor(b,lat,strong=150.0):
    _,_,h,w=b.shape
    yy=torch.linspace(-1,1,h).view(1,h,1); xx=torch.linspace(-1,1,w).view(1,1,w)
    rows=[]
    for i in range(len(b)):
        z=b[i,0]
        pw=torch.where(z>=strong,z,torch.zeros_like(z)); nw=torch.where(z<=-strong,-z,torch.zeros_like(z))
        ps=float(pw.sum()); ns=float(nw.sum())
        if ps<=0 or ns<=0:
            rows.append([0.,0.,0.,0.,0.]); continue
        px=float((pw*xx).sum()/pw.sum()); py=float((pw*yy).sum()/pw.sum())
        nx=float((nw*xx).sum()/nw.sum()); ny=float((nw*yy).sum()/nw.sum())
        dx=px-nx; dy=py-ny; sep=max((dx*dx+dy*dy)**0.5,1e-8); hemi=1. if lat[i]>=0 else -1.
        rows.append([hemi*dx/sep,hemi*dy/sep,np.log1p(sep),np.log1p((ps+ns)/(h*w)),1.0])
    return np.asarray(rows,np.float32)


def generic_descriptor(b: torch.Tensor) -> np.ndarray:
    """Generic image statistics deliberately independent of HJ/PIL target metrics."""
    a=b.abs()
    flat=a.flatten(1)
    mean_abs=flat.mean(1)
    p90=torch.quantile(flat,0.90,dim=1)
    p99=torch.quantile(flat,0.99,dim=1)
    active=(a>150).float().mean((1,2,3))
    strong=(a>500).float().mean((1,2,3))
    sat=(a>2900).float().mean((1,2,3))
    return torch.stack([
        torch.log1p(mean_abs/50.0),torch.log1p(p90/50.0),torch.log1p(p99/50.0),
        active,strong,sat,
    ],dim=1).cpu().numpy().astype(np.float32)


def pooled_vectors(b: torch.Tensor) -> torch.Tensor:
    z=torch.asinh(b/300.)/torch.asinh(torch.tensor(10.,dtype=b.dtype,device=b.device))
    return F.adaptive_avg_pool2d(z,(16,16)).flatten(1)


def scalar_summary(x):
    x=np.asarray(x,float)
    return {'mean':float(np.nanmean(x)),'median':float(np.nanmedian(x)),'p10':float(np.nanpercentile(x,10)),'p90':float(np.nanpercentile(x,90))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True)
    ap.add_argument('--synthetic-manifest',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--real-per-group',type=int,default=3); ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    real,rlat,rrec=load_real(args.evidence_dir,args.cache_dir,args.real_per_group,args.seed)
    syn,slat,sman=load_synthetic(args.synthetic_manifest)
    rp=hard_pil_descriptor(real); sp=hard_pil_descriptor(syn)
    rg=hard_geometry_descriptor(real,rlat); sg=hard_geometry_descriptor(syn,slat)
    rgen=generic_descriptor(real); sgen=generic_descriptor(syn)

    rps,sps,pmed,pscale=robust_standardize(rp,sp); rgs,sgs,gmed,gscale=robust_standardize(rg,sg)
    rgn,sgn,genmed,genscale=robust_standardize(rgen,sgen)
    pil_dist=energy_distance(torch.from_numpy(sps),torch.from_numpy(rps))
    geom_dist=energy_distance(torch.from_numpy(sgs),torch.from_numpy(rgs))
    generic_dist=energy_distance(torch.from_numpy(sgn),torch.from_numpy(rgn))
    generic_rr=split_half_baseline(rgn,args.seed+71)

    real_flux=real.abs().mean((1,2,3)).numpy(); syn_flux=syn.abs().mean((1,2,3)).numpy()
    real_active=(real.abs()>150).float().mean((1,2,3)).numpy(); syn_active=(syn.abs()>150).float().mean((1,2,3)).numpy()
    real_pooled=pooled_vectors(real); syn_pooled=pooled_vectors(syn)
    real_div=float(torch.pdist(real_pooled[:min(256,len(real_pooled))]).mean()) if len(real_pooled)>1 else float('nan')
    syn_div=float(torch.pdist(syn_pooled[:min(256,len(syn_pooled))]).mean()) if len(syn_pooled)>1 else float('nan')
    nn_real=float(torch.cdist(syn_pooled[:min(256,len(syn_pooled))],real_pooled[:min(512,len(real_pooled))]).min(1).values.mean()) if len(syn_pooled) and len(real_pooled) else float('nan')

    real_flux_med=float(np.nanmedian(real_flux)); syn_flux_med=float(np.nanmedian(syn_flux))
    real_active_med=float(np.nanmedian(real_active)); syn_active_med=float(np.nanmedian(syn_active))
    flux_ratio=syn_flux_med/max(real_flux_med,EPS)
    active_ratio=syn_active_med/max(real_active_med,EPS)
    diversity_ratio=syn_div/max(real_div,EPS) if np.isfinite(real_div) else float('nan')
    sat_frac=float((syn.abs()>2900).float().mean())
    rr_ref=max(generic_rr['p90'],generic_rr['median'],1e-6)
    generic_distance_ratio=generic_dist/rr_ref

    # Predeclared quality floor. It is intentionally broad: this gate rejects
    # obviously untrained/noisy generators, not stylistic differences from real data.
    generic_gate_pass=bool(
        0.50 <= flux_ratio <= 2.00 and
        0.50 <= active_ratio <= 2.00 and
        0.40 <= diversity_ratio <= 2.50 and
        sat_frac < 0.01 and
        generic_distance_ratio <= 8.0
    )

    pil_cols=['mean_grad','rms_grad','top10_grad','frac_gt100','frac_gt250','frac_gt500','pil_area_fraction','has_pil']
    geom_cols=['hemi_ux','hemi_uy','log_sep','log_strong_flux_density','has_bipole']
    gen_cols=['log_mean_abs','log_p90_abs','log_p99_abs','active_fraction','strong_fraction','saturation_fraction']
    pd.DataFrame(sp,columns=pil_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out/'synthetic_hard_pil.csv',index=False)
    pd.DataFrame(rp,columns=pil_cols).assign(sample_id=rrec.sample_id.astype(str).values,region_group_id=rrec.region_group_id.astype(str).values).to_csv(out/'real_hard_pil.csv',index=False)
    pd.DataFrame(sg,columns=geom_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out/'synthetic_hard_geometry.csv',index=False)
    pd.DataFrame(rg,columns=geom_cols).assign(sample_id=rrec.sample_id.astype(str).values).to_csv(out/'real_hard_geometry.csv',index=False)
    pd.DataFrame(sgen,columns=gen_cols).assign(synthetic_id=sman.synthetic_id.astype(str).values).to_csv(out/'synthetic_generic.csv',index=False)
    pd.DataFrame(rgen,columns=gen_cols).assign(sample_id=rrec.sample_id.astype(str).values,region_group_id=rrec.region_group_id.astype(str).values).to_csv(out/'real_generic.csv',index=False)

    report={
        'evaluator':'v2 independent hard thresholds; not the differentiable training proxy',
        'real_count':len(real),'synthetic_count':len(syn),
        'hard_pil_standardized_energy_distance':pil_dist,
        'hard_geometry_standardized_energy_distance':geom_dist,
        'generic_standardized_energy_distance':generic_dist,
        'generic_real_split_half_distance':generic_rr,
        'generic_distance_to_real_baseline_ratio':generic_distance_ratio,
        'real_pil_mean_gradient_g_per_mm':scalar_summary(rp[:,0]),
        'synthetic_pil_mean_gradient_g_per_mm':scalar_summary(sp[:,0]),
        'real_pil_top10_gradient_g_per_mm':scalar_summary(rp[:,2]),
        'synthetic_pil_top10_gradient_g_per_mm':scalar_summary(sp[:,2]),
        'real_unsigned_flux_proxy':scalar_summary(real_flux),'synthetic_unsigned_flux_proxy':scalar_summary(syn_flux),
        'real_strong_field_area_fraction':scalar_summary(real_active),'synthetic_strong_field_area_fraction':scalar_summary(syn_active),
        'synthetic_to_real_flux_median_ratio':flux_ratio,
        'synthetic_to_real_active_area_median_ratio':active_ratio,
        'real_coarse_pairwise_diversity':real_div,
        'synthetic_coarse_pairwise_diversity':syn_div,
        'synthetic_to_real_diversity_ratio':diversity_ratio,
        'synthetic_mean_nearest_real_coarse_distance':nn_real,
        'synthetic_saturation_fraction_abs_gt_2900G':sat_frac,
        'generic_fidelity_gate_pass':generic_gate_pass,
        'generic_fidelity_gate_definition':{
            'flux_median_ratio':[0.50,2.00],
            'active_area_median_ratio':[0.50,2.00],
            'diversity_ratio':[0.40,2.50],
            'saturation_fraction_max':0.01,
            'generic_distance_to_real_split_p90_max':8.0,
        },
        'robust_scaling':{
            'pil_median':pmed.tolist(),'pil_scale':pscale.tolist(),
            'geometry_median':gmed.tolist(),'geometry_scale':gscale.tolist(),
            'generic_median':genmed.tolist(),'generic_scale':genscale.tolist(),
        },
    }
    (out/'v2_manipulation_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    print(json.dumps(report,indent=2,allow_nan=True))


if __name__=='__main__':
    main()
