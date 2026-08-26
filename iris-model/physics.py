from __future__ import annotations

import torch
import torch.nn.functional as F


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable multivariate energy distance between two batches."""
    x=x.reshape(x.shape[0],-1); y=y.reshape(y.shape[0],-1)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return torch.mean((x.mean(0)-y.mean(0))**2)
    xy=torch.cdist(x,y,p=2).mean()
    xx=torch.cdist(x,x,p=2).mean()
    yy=torch.cdist(y,y,p=2).mean()
    return 2*xy-xx-yy


def polarity_geometry_descriptor(b: torch.Tensor, latitude_deg: torch.Tensor, temperature_g: float = 120.0) -> torch.Tensor:
    """Soft bipole geometry descriptor used for the Hale/Joy population loss.

    Rather than hard-coding an image-axis convention, we match generated descriptor
    distributions to REAL training magnetograms conditioned on the same hemisphere.
    This makes the constraint invariant to the dataset's stored east/west convention.
    """
    if b.ndim==3: b=b[:,None]
    n,_,h,w=b.shape
    yy=torch.linspace(-1,1,h,device=b.device,dtype=b.dtype).view(1,1,h,1)
    xx=torch.linspace(-1,1,w,device=b.device,dtype=b.dtype).view(1,1,1,w)
    pos=F.softplus(b/temperature_g)
    neg=F.softplus(-b/temperature_g)
    eps=1e-6
    px=(pos*xx).sum((2,3))/(pos.sum((2,3))+eps); py=(pos*yy).sum((2,3))/(pos.sum((2,3))+eps)
    nx=(neg*xx).sum((2,3))/(neg.sum((2,3))+eps); ny=(neg*yy).sum((2,3))/(neg.sum((2,3))+eps)
    dx=(px-nx).squeeze(1); dy=(py-ny).squeeze(1)
    sep=torch.sqrt(dx*dx+dy*dy+eps)
    # Hemisphere sign is supplied as conditioning metadata. Matching (dx,dy) within
    # the same hemisphere captures the population polarity ordering and Joy tilt.
    hemi=torch.sign(latitude_deg).to(b.dtype)
    return torch.stack([hemi*dx, hemi*dy, sep],dim=1)


def _neighbor_mean(x: torch.Tensor) -> torch.Tensor:
    k=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],device=x.device,dtype=x.dtype).view(1,1,3,3)/4.0
    return F.conv2d(x,k,padding=1)


def soft_pil_gradient_score(
    b: torch.Tensor,
    pixel_mm: float = 2.0,
    strong_field_g: float = 150.0,
    membership_temp_g: float = 60.0,
) -> torch.Tensor:
    """Differentiable strong-gradient PIL statistic in G/Mm.

    Opposite-polarity contact is represented with smooth positive/negative field
    memberships. This avoids the sign error that occurs when sigmoid(+Bi*Bj/T) is
    used for a polarity-inversion weight.
    """
    if b.ndim==3: b=b[:,None]
    pos=torch.sigmoid((b-strong_field_g)/membership_temp_g)
    neg=torch.sigmoid((-b-strong_field_g)/membership_temp_g)
    contact=pos*_neighbor_mean(neg)+neg*_neighbor_mean(pos)
    # central differences; denominator is 2*pixel spacing
    gx=(F.pad(b,(1,1,0,0),mode='replicate')[:,:,:,2:]-F.pad(b,(1,1,0,0),mode='replicate')[:,:,:,:-2])/(2*pixel_mm)
    gy=(F.pad(b,(0,0,1,1),mode='replicate')[:,:,2:,:]-F.pad(b,(0,0,1,1),mode='replicate')[:,:,:-2,:])/(2*pixel_mm)
    grad=torch.sqrt(gx*gx+gy*gy+1e-6)
    score=(contact*grad).sum((1,2,3))/(contact.sum((1,2,3))+1e-6)
    return score


def population_loss(fake_b: torch.Tensor, real_b: torch.Tensor, latitude_deg: torch.Tensor) -> torch.Tensor:
    # Split by hemisphere so north/south populations cannot cancel each other.
    losses=[]
    for mask in [latitude_deg>=0,latitude_deg<0]:
        if int(mask.sum())>=2:
            f=polarity_geometry_descriptor(fake_b[mask],latitude_deg[mask])
            r=polarity_geometry_descriptor(real_b[mask].detach(),latitude_deg[mask])
            losses.append(energy_distance(f,r))
    return torch.stack(losses).mean() if losses else fake_b.sum()*0.0


def pil_distribution_loss(fake_b: torch.Tensor, real_b: torch.Tensor, pixel_mm: float = 2.0) -> torch.Tensor:
    f=soft_pil_gradient_score(fake_b,pixel_mm).unsqueeze(1)
    r=soft_pil_gradient_score(real_b.detach(),pixel_mm).unsqueeze(1)
    return energy_distance(f,r)
