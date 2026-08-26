from __future__ import annotations

import json
import torch
import torch.nn.functional as F

from physics_v2 import (
    soft_pil_contact,
    strong_pil_gradient_descriptor_v2,
    pil_distribution_loss_v2,
    polarity_geometry_descriptor_v2,
    population_distribution_loss_v2,
)


def make_bipole(batch=8, size=128, sharp=True, device='cpu'):
    y, x = torch.meshgrid(torch.linspace(-1,1,size,device=device), torch.linspace(-1,1,size,device=device), indexing='ij')
    fields=[]
    for i in range(batch):
        tilt=(i-(batch-1)/2)*0.02
        p=torch.exp(-((x+0.25)**2+(y-tilt)**2)/(2*0.12**2))
        n=torch.exp(-((x-0.25)**2+(y+tilt)**2)/(2*0.12**2))
        b=1800*(p-n)
        if not sharp:
            b=F.avg_pool2d(b[None,None],kernel_size=9,stride=1,padding=4)[0,0]
        fields.append(b)
    return torch.stack(fields)[:,None]


def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    zero=torch.zeros(4,1,128,128,device=device)
    zero_contact=float(soft_pil_contact(zero).mean().item())
    if zero_contact > 1e-5:
        raise AssertionError(f'quiet-field PIL contact too large: {zero_contact}')

    sharp=make_bipole(device=device,sharp=True)
    blur=make_bipole(device=device,sharp=False)
    ds=strong_pil_gradient_descriptor_v2(sharp)
    db=strong_pil_gradient_descriptor_v2(blur)
    # RMS/tail gradient descriptors must respond to selective PIL blurring.
    if not float(ds[:,1].mean()) > float(db[:,1].mean()):
        raise AssertionError('PIL RMS descriptor does not distinguish sharp from blurred bipoles')
    if not float(ds[:,2].mean()) > float(db[:,2].mean()):
        raise AssertionError('PIL tail descriptor does not distinguish sharp from blurred bipoles')

    fake=blur.clone().requires_grad_(True)
    lp=pil_distribution_loss_v2(fake,sharp)
    lp.backward()
    grad_norm=float(fake.grad.abs().mean().item())
    if not torch.isfinite(fake.grad).all() or grad_norm <= 0:
        raise AssertionError('PIL distribution loss has invalid/zero gradient')

    lat=torch.tensor([20.,18.,15.,12.,-12.,-15.,-18.,-20.],device=device)
    geom=polarity_geometry_descriptor_v2(sharp,lat)
    flipped=torch.flip(sharp,dims=[3]).clone().requires_grad_(True)
    lg=population_distribution_loss_v2(flipped,sharp,lat)
    lg.backward()
    geom_delta=float((polarity_geometry_descriptor_v2(flipped.detach(),lat)-geom).abs().mean().item())
    geom_grad=float(flipped.grad.abs().mean().item())
    if geom_delta <= 1e-3 or geom_grad <= 0:
        raise AssertionError('geometry descriptor/loss does not respond to polarity geometry change')

    report={
        'device':device,
        'quiet_field_contact_mean':zero_contact,
        'sharp_pil_descriptor_mean':ds.mean(0).detach().cpu().tolist(),
        'blurred_pil_descriptor_mean':db.mean(0).detach().cpu().tolist(),
        'pil_loss_blur_to_sharp':float(lp.item()),
        'pil_gradient_mean_abs':grad_norm,
        'geometry_flip_descriptor_delta':geom_delta,
        'geometry_loss':float(lg.item()),
        'geometry_gradient_mean_abs':geom_grad,
        'status':'PASS',
    }
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
