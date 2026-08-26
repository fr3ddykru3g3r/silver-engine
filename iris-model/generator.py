from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half=dim//2
    freqs=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/max(1,half-1))
    args=t.float()[:,None]*freqs[None]
    emb=torch.cat([torch.sin(args),torch.cos(args)],dim=1)
    if dim%2: emb=F.pad(emb,(0,1))
    return emb


def groups(c: int) -> int:
    g=min(8,c)
    while c%g:g-=1
    return g


class ResBlock(nn.Module):
    def __init__(self,cin,cout,emb_dim,dropout=0.0):
        super().__init__()
        self.n1=nn.GroupNorm(groups(cin),cin); self.c1=nn.Conv2d(cin,cout,3,padding=1)
        self.e=nn.Linear(emb_dim,cout*2)
        self.n2=nn.GroupNorm(groups(cout),cout); self.c2=nn.Conv2d(cout,cout,3,padding=1)
        self.drop=nn.Dropout(dropout)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x,emb):
        h=self.c1(F.silu(self.n1(x)))
        scale,shift=self.e(F.silu(emb)).chunk(2,dim=1)
        h=self.n2(h)*(1+scale[:,:,None,None])+shift[:,:,None,None]
        h=self.c2(self.drop(F.silu(h)))
        return h+self.skip(x)


class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.n=nn.GroupNorm(groups(c),c); self.qkv=nn.Conv2d(c,c*3,1); self.proj=nn.Conv2d(c,c,1)
    def forward(self,x):
        b,c,h,w=x.shape; z=self.qkv(self.n(x)); q,k,v=z.chunk(3,dim=1)
        q=q.reshape(b,c,h*w).transpose(1,2); k=k.reshape(b,c,h*w); v=v.reshape(b,c,h*w).transpose(1,2)
        a=torch.softmax(torch.bmm(q,k)/math.sqrt(c),dim=-1)
        y=torch.bmm(a,v).transpose(1,2).reshape(b,c,h,w)
        return x+self.proj(y)


class ConditionalUNet(nn.Module):
    """128x128 one-channel DDPM U-Net conditioned on flare label and latitude."""
    def __init__(self,base=48,emb_dim=192,dropout=0.05):
        super().__init__(); self.emb_dim=emb_dim
        self.time_mlp=nn.Sequential(nn.Linear(emb_dim,emb_dim*4),nn.SiLU(),nn.Linear(emb_dim*4,emb_dim))
        self.cond_mlp=nn.Sequential(nn.Linear(2,emb_dim),nn.SiLU(),nn.Linear(emb_dim,emb_dim))
        self.inp=nn.Conv2d(1,base,3,padding=1)
        self.d1a=ResBlock(base,base,emb_dim,dropout); self.d1b=ResBlock(base,base,emb_dim,dropout)
        self.ds1=nn.Conv2d(base,base*2,4,stride=2,padding=1)
        self.d2a=ResBlock(base*2,base*2,emb_dim,dropout); self.d2b=ResBlock(base*2,base*2,emb_dim,dropout)
        self.ds2=nn.Conv2d(base*2,base*4,4,stride=2,padding=1)
        self.d3a=ResBlock(base*4,base*4,emb_dim,dropout); self.d3b=ResBlock(base*4,base*4,emb_dim,dropout)
        self.ds3=nn.Conv2d(base*4,base*4,4,stride=2,padding=1)
        self.mid1=ResBlock(base*4,base*4,emb_dim,dropout); self.attn=Attention(base*4); self.mid2=ResBlock(base*4,base*4,emb_dim,dropout)
        self.us3=nn.ConvTranspose2d(base*4,base*4,4,stride=2,padding=1)
        self.u3a=ResBlock(base*8,base*4,emb_dim,dropout); self.u3b=ResBlock(base*4,base*4,emb_dim,dropout)
        self.us2=nn.ConvTranspose2d(base*4,base*2,4,stride=2,padding=1)
        self.u2a=ResBlock(base*4,base*2,emb_dim,dropout); self.u2b=ResBlock(base*2,base*2,emb_dim,dropout)
        self.us1=nn.ConvTranspose2d(base*2,base,4,stride=2,padding=1)
        self.u1a=ResBlock(base*2,base,emb_dim,dropout); self.u1b=ResBlock(base,base,emb_dim,dropout)
        self.out=nn.Sequential(nn.GroupNorm(groups(base),base),nn.SiLU(),nn.Conv2d(base,1,3,padding=1))
    def embed(self,t,label,latitude):
        te=timestep_embedding(t,self.emb_dim); te=self.time_mlp(te)
        c=torch.stack([label.float(),torch.clamp(latitude.float()/40.0,-1,1)],dim=1)
        return te+self.cond_mlp(c)
    def forward(self,x,t,label,latitude):
        e=self.embed(t,label,latitude); x0=self.inp(x)
        a=self.d1b(self.d1a(x0,e),e); b=self.d2b(self.d2a(self.ds1(a),e),e); c=self.d3b(self.d3a(self.ds2(b),e),e)
        m=self.mid2(self.attn(self.mid1(self.ds3(c),e)),e)
        z=self.us3(m); z=self.u3b(self.u3a(torch.cat([z,c],1),e),e)
        z=self.us2(z); z=self.u2b(self.u2a(torch.cat([z,b],1),e),e)
        z=self.us1(z); z=self.u1b(self.u1a(torch.cat([z,a],1),e),e)
        return self.out(z)


def cosine_betas(steps: int, s: float = 0.008):
    x=torch.linspace(0,steps,steps+1)
    ac=torch.cos(((x/steps)+s)/(1+s)*math.pi*0.5)**2; ac=ac/ac[0]
    b=1-(ac[1:]/ac[:-1]); return torch.clamp(b,1e-5,0.999)


class Diffusion:
    def __init__(self,steps=400,device='cpu'):
        self.steps=steps; self.device=torch.device(device)
        b=cosine_betas(steps).to(self.device); a=1-b; ab=torch.cumprod(a,0)
        self.b=b; self.a=a; self.ab=ab; self.sqrt_ab=torch.sqrt(ab); self.sqrt_om=torch.sqrt(1-ab)
    def q_sample(self,x0,t,noise=None):
        if noise is None:noise=torch.randn_like(x0)
        return self.sqrt_ab[t][:,None,None,None]*x0+self.sqrt_om[t][:,None,None,None]*noise,noise
    def x0_from_eps(self,xt,t,eps):
        return (xt-self.sqrt_om[t][:,None,None,None]*eps)/(self.sqrt_ab[t][:,None,None,None]+1e-8)
    @torch.no_grad()
    def sample(self,model,n,label,latitude,shape=(1,128,128)):
        x=torch.randn((n,*shape),device=self.device)
        for i in reversed(range(self.steps)):
            t=torch.full((n,),i,device=self.device,dtype=torch.long); eps=model(x,t,label,latitude)
            alpha=self.a[i]; abar=self.ab[i]; beta=self.b[i]
            mean=(x-(beta/torch.sqrt(1-abar))*eps)/torch.sqrt(alpha)
            x=mean+(torch.sqrt(beta)*torch.randn_like(x) if i>0 else 0)
        return torch.clamp(x,-1,1)
    @torch.no_grad()
    def ddim_sample(self,model,n,label,latitude,shape=(1,128,128),sampling_steps=50,eta=0.0):
        """Accelerated DDIM sampler using a subsequence of the training schedule."""
        sampling_steps=max(2,min(int(sampling_steps),self.steps))
        seq=torch.linspace(0,self.steps-1,sampling_steps,device=self.device).round().long().unique()
        x=torch.randn((n,*shape),device=self.device)
        for j in reversed(range(len(seq))):
            i=int(seq[j].item()); t=torch.full((n,),i,device=self.device,dtype=torch.long)
            eps=model(x,t,label,latitude)
            abar=self.ab[i]
            x0=torch.clamp((x-torch.sqrt(1-abar)*eps)/(torch.sqrt(abar)+1e-8),-1,1)
            if j==0:
                x=x0; continue
            ip=int(seq[j-1].item()); abar_prev=self.ab[ip]
            sigma=eta*torch.sqrt(torch.clamp((1-abar_prev)/(1-abar)*(1-abar/abar_prev),min=0.0))
            direction=torch.sqrt(torch.clamp(1-abar_prev-sigma*sigma,min=0.0))*eps
            noise=torch.randn_like(x) if eta>0 else 0.0
            x=torch.sqrt(abar_prev)*x0+direction+sigma*noise
        return torch.clamp(x,-1,1)
