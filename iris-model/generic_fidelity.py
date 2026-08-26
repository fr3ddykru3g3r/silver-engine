from __future__ import annotations

import torch
import torch.nn.functional as F

EPS=1e-6


def _quantile_loss(x:torch.Tensor,y:torch.Tensor,qs=(0.1,0.25,0.5,0.75,0.9)):
    q=torch.tensor(qs,device=x.device,dtype=x.dtype)
    return F.smooth_l1_loss(torch.quantile(x,q,dim=0),torch.quantile(y.detach(),q,dim=0))


def generic_descriptor(b:torch.Tensor,strong_field_g:float=150.0,temp_g:float=40.0):
    if b.ndim==3:b=b[:,None]
    absb=b.abs()
    mean_abs=absb.mean((1,2,3))
    rms=torch.sqrt((b.square()).mean((1,2,3))+EPS)
    strong=torch.sigmoid((absb-strong_field_g)/temp_g).mean((1,2,3))
    very_strong=torch.sigmoid((absb-1000.0)/80.0).mean((1,2,3))
    sat=torch.sigmoid((absb-2800.0)/40.0).mean((1,2,3))
    pos=F.relu(b-strong_field_g).sum((1,2,3)); neg=F.relu(-b-strong_field_g).sum((1,2,3))
    imbalance=(pos-neg).abs()/(pos+neg+EPS)
    return torch.stack([
        torch.log1p(mean_abs/50.0),
        torch.log1p(rms/50.0),
        strong,
        very_strong,
        sat,
        imbalance,
    ],dim=1)


def generic_distribution_loss(fake_b:torch.Tensor,real_b:torch.Tensor):
    f=generic_descriptor(fake_b);r=generic_descriptor(real_b.detach())
    # batch-wide energy-style term + robust quantiles
    xy=torch.cdist(f,r).mean();xx=torch.cdist(f,f).mean();yy=torch.cdist(r,r).mean()
    energy=2*xy-xx-yy
    return 0.5*energy+0.5*_quantile_loss(f,r)
