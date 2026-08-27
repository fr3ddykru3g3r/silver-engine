from __future__ import annotations

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cvae import MagnetogramCVAE
from data import build_records, cache_records, MagnetogramDataset
from train_cvae_physics import positive_subset


def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True); ap.add_argument('--evidence-dir',required=True); ap.add_argument('--cache-dir',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--per-source-group',type=int,default=4); ap.add_argument('--max-groups',type=int,default=0)
    ap.add_argument('--latent-bank-per-group',type=int,default=4); ap.add_argument('--noise-scale',type=float,default=0.20)
    ap.add_argument('--cross-region-mix',type=float,default=0.12); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--download-workers',type=int,default=12)
    args=ap.parse_args(); seed_all(args.seed)
    rng=np.random.default_rng(args.seed)
    out=Path(args.out_dir); arrdir=out/'arrays'; arrdir.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MagnetogramCVAE(int(ck['base']),int(ck['latent_dim'])).to(device); model.load_state_dict(ck['model']); model.eval()

    rec=positive_subset(build_records(args.evidence_dir,'train'),args.latent_bank_per_group,args.seed)
    if args.max_groups and rec.region_group_id.nunique()>args.max_groups:
        keep=(rec[['region_group_id']].drop_duplicates().sample(n=args.max_groups,random_state=args.seed).region_group_id.astype(str).tolist())
        rec=rec[rec.region_group_id.astype(str).isin(keep)].copy()
    rec=cache_records(rec,Path(args.cache_dir),workers=args.download_workers)
    ds=MagnetogramDataset(rec); dl=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    mus=[]; lvs=[]; lats=[]; groups=[]; sids=[]
    with torch.no_grad():
        offset=0
        for b in dl:
            x=b['x'].to(device); mu,lv=model.encode(x)
            n=len(x); mus.append(mu.cpu()); lvs.append(lv.cpu()); lats.extend([float(v) for v in b['latitude']]); groups.extend([str(v) for v in b['group']]); sids.extend([str(v) for v in b['sample_id']]); offset+=n
    mu=torch.cat(mus); lv=torch.cat(lvs); lat=np.asarray(lats,np.float32); groups=np.asarray(groups,str); sids=np.asarray(sids,str)
    latent_scale=torch.std(mu,dim=0,unbiased=False).clamp_min(0.05)
    unique=sorted(set(groups.tolist()))
    rows=[]; serial=0
    for gid in unique:
        idx=np.where(groups==gid)[0]
        glat=float(np.median(lat[idx])); hemi=1 if glat>=0 else -1
        pool=np.where((np.sign(lat)==hemi)&(groups!=gid))[0]
        if len(pool)==0: pool=np.where(groups!=gid)[0]
        # Prefer nearby latitude so hemisphere/population geometry is not forced to
        # interpolate between physically unrelated latitudes.
        if len(pool):
            d=np.abs(lat[pool]-glat); pool=pool[np.argsort(d)[:min(64,len(pool))]]
        for _ in range(args.per_source_group):
            i=int(rng.choice(idx)); j=int(rng.choice(pool)) if len(pool) else i
            m=float(rng.uniform(-args.cross_region_mix,args.cross_region_mix))
            eps=torch.from_numpy(rng.standard_normal(mu.shape[1]).astype(np.float32))
            z=mu[i] + m*(mu[j]-mu[i]) + args.noise_scale*latent_scale*eps
            with torch.no_grad():
                x=model.decode(z[None].to(device),torch.tensor([glat],device=device)).cpu().numpy()[0,0].astype(np.float32)
            sid=f"cvae_{ck.get('condition','base')}_{args.seed}_{gid}_{serial:06d}"; serial+=1
            p=arrdir/f'{sid}.npy'; np.save(p,x)
            rows.append({'synthetic_id':sid,'array_path':str(p.resolve()),'source_region_group_id':gid,'latitude_deg':glat,'label_m1plus_24h':1,'generator_condition':ck.get('condition','base'),'generator_family':'cvae_v3','generator_seed':args.seed,'source_sample_id':sids[i],'partner_sample_id':sids[j],'latent_noise_scale':args.noise_scale,'cross_region_mix':m})
    man=pd.DataFrame(rows); man.to_csv(out/'synthetic_manifest.csv',index=False)
    summary={'generator_family':'cvae_v3','condition':ck.get('condition','base'),'source_groups':len(unique),'synthetic_count':len(man),'per_source_group':args.per_source_group,'latent_noise_scale':args.noise_scale,'cross_region_mix_max':args.cross_region_mix,'seed':args.seed,'device':str(device)}
    (out/'sampling_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
