from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import build_records,cache_records,MagnetogramDataset
from train_generator import group_temporal_subset
from preprocess import denormalize_gauss
from physics import polarity_geometry_descriptor,soft_pil_gradient_score,energy_distance


def load_real(evidence_dir,cache_dir,per_group,seed,device):
    rec=build_records(evidence_dir,'train');rec=rec[rec.label_m1plus_24h.eq(1)].copy();rec=group_temporal_subset(rec,per_group,per_group,seed);rec=cache_records(rec,cache_dir,workers=12);ds=MagnetogramDataset(rec);raws=[];lats=[]
    for i in range(len(ds)):
        z=ds[i];raws.append(z['raw_gauss']);lats.append(z['latitude'])
    return torch.stack(raws).to(device),torch.stack(lats).to(device),rec

def load_synthetic(manifest,device):
    p=Path(manifest);m=pd.read_csv(p);arr=[]
    for raw in m.array_path:
        q=Path(str(raw))
        if not q.exists():q=p.parent/'arrays'/q.name
        arr.append(np.load(q).astype(np.float32))
    x=torch.from_numpy(np.stack(arr))[:,None].to(device);return denormalize_gauss(x),torch.tensor(m.latitude_deg.to_numpy(dtype=np.float32),device=device),m

def pooled_vectors(b):
    z=torch.asinh(b/300.)/torch.asinh(torch.tensor(10.,device=b.device,dtype=b.dtype));return F.adaptive_avg_pool2d(z,(16,16)).flatten(1)
def summary_stats(x):
    a=x.detach().cpu().numpy();return {'mean':float(np.mean(a)),'median':float(np.median(a)),'std':float(np.std(a)),'p10':float(np.percentile(a,10)),'p90':float(np.percentile(a,90))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',required=True);ap.add_argument('--cache-dir',required=True);ap.add_argument('--synthetic-manifest',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--real-per-group',type=int,default=4);ap.add_argument('--seed',type=int,default=2026);args=ap.parse_args();out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');real,rlat,rrec=load_real(args.evidence_dir,args.cache_dir,args.real_per_group,args.seed,device);syn,slat,sman=load_synthetic(args.synthetic_manifest,device)
    rp=soft_pil_gradient_score(real,2.);sp=soft_pil_gradient_score(syn,2.);rg=polarity_geometry_descriptor(real,rlat);sg=polarity_geometry_descriptor(syn,slat);pil_ed=float(energy_distance(torch.log1p(sp/50)[:,None],torch.log1p(rp/50)[:,None]).item());geom_ed=float(energy_distance(sg,rg).item());sv=pooled_vectors(syn);rv=pooled_vectors(real);ssub=sv[:min(256,len(sv))];rsub=rv[:min(512,len(rv))];diversity=float(torch.pdist(ssub).mean().item()) if len(ssub)>1 else float('nan');nn_real=float(torch.cdist(ssub,rsub).min(dim=1).values.mean().item()) if len(ssub) and len(rsub) else float('nan')
    np.savez_compressed(out/'physics_sample_arrays.npz',synthetic_pil=sp.detach().cpu().numpy(),real_pil=rp.detach().cpu().numpy(),synthetic_geometry=sg.detach().cpu().numpy(),real_geometry=rg.detach().cpu().numpy(),synthetic_latitude=slat.detach().cpu().numpy(),real_latitude=rlat.detach().cpu().numpy())
    pd.DataFrame({'synthetic_id':sman.synthetic_id.astype(str),'source_region_group_id':sman.source_region_group_id.astype(str),'pil_gradient_score':sp.detach().cpu().numpy(),'latitude_deg':slat.detach().cpu().numpy()}).to_csv(out/'synthetic_physics_rows.csv',index=False)
    pd.DataFrame({'sample_id':rrec.sample_id.astype(str),'region_group_id':rrec.region_group_id.astype(str),'pil_gradient_score':rp.detach().cpu().numpy(),'latitude_deg':rlat.detach().cpu().numpy()}).to_csv(out/'real_physics_rows.csv',index=False)
    report={'device':str(device),'synthetic_count':len(sman),'real_reference_count':len(rrec),'pil_real':summary_stats(rp),'pil_synthetic':summary_stats(sp),'pil_log_energy_distance':pil_ed,'geometry_energy_distance':geom_ed,'synthetic_coarse_pairwise_diversity':diversity,'synthetic_mean_nearest_real_coarse_distance':nn_real,'synthetic_abs_gt_2900G_fraction':float((syn.abs()>2900).float().mean().item()),'sample_level_diagnostics_saved':True}
    (out/'generator_physics_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n');print(json.dumps(report,indent=2,allow_nan=True),flush=True)
if __name__=='__main__':main()
