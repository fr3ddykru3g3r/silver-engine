from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_array(path, manifest_dir):
    q=Path(str(path))
    if not q.exists(): q=manifest_dir/'arrays'/q.name
    return np.load(q).astype(np.float32)


def pil_blur(x: np.ndarray, kernel=9):
    """Selectively smooth only a narrow polarity-contact neighbourhood.

    This destroys extreme PIL gradients while leaving most coarse magnetic structure
    unchanged. Operates in normalized generator space, not Gauss.
    """
    t=torch.from_numpy(x)[None,None]
    # Sign-contact proxy on normalized field; dilation identifies nearby opposite signs.
    pos=(t>0.08).float(); neg=(t<-0.08).float()
    dp=F.max_pool2d(pos,5,1,2); dn=F.max_pool2d(neg,5,1,2)
    contact=((dp>0.5)&(dn>0.5)).float()
    smooth=F.avg_pool2d(t,kernel,1,kernel//2)
    # feather the binary region one extra pixel to avoid a hard seam
    w=F.max_pool2d(contact,3,1,1)
    return (t*(1-w)+smooth*w)[0,0].numpy().astype(np.float32)


def geometry_flip(x: np.ndarray):
    """Reverse east-west magnetic geometry while preserving pixel-value histogram."""
    return np.flip(x,axis=1).copy().astype(np.float32)


def spatial_shuffle_blocks(x: np.ndarray, block=16, seed=0):
    """Destroy large-scale organization while preserving local block texture/histogram."""
    rng=np.random.default_rng(seed)
    h,w=x.shape
    blocks=[]
    for y in range(0,h,block):
        for xx in range(0,w,block):
            blocks.append(x[y:y+block,xx:xx+block].copy())
    order=rng.permutation(len(blocks)); out=np.empty_like(x); k=0
    for y in range(0,h,block):
        for xx in range(0,w,block):
            out[y:y+block,xx:xx+block]=blocks[order[k]]; k+=1
    return out.astype(np.float32)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,default=2026); args=ap.parse_args()
    p=Path(args.manifest); m=pd.read_csv(p); root=p.parent; out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    controls=['pil_blur','geometry_flip','block_shuffle']
    counts={}
    for control in controls:
        arrdir=out/control/'arrays'; arrdir.mkdir(parents=True,exist_ok=True); rows=[]
        for i,r in m.iterrows():
            x=load_array(r.array_path,root)
            if control=='pil_blur': z=pil_blur(x)
            elif control=='geometry_flip': z=geometry_flip(x)
            else: z=spatial_shuffle_blocks(x,seed=args.seed+i)
            sid=f"{r.synthetic_id}__{control}"; q=arrdir/f'{sid}.npy'; np.save(q,z)
            d=r.to_dict(); d.update({'synthetic_id':sid,'array_path':str(q.resolve()),'destructive_control':control,'parent_synthetic_id':str(r.synthetic_id)})
            rows.append(d)
        man=pd.DataFrame(rows); man.to_csv(out/control/'synthetic_manifest.csv',index=False); counts[control]=len(man)
    report={'source_manifest':str(p),'seed':args.seed,'controls':counts,
            'purpose':'counterfactual controls: selectively remove magnetic organization while preserving as much nuisance distribution as possible; must be evaluated with manipulation checks before downstream use'}
    (out/'control_summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
