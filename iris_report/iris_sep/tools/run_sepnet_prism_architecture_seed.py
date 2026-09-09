"""Same-cohort released-architecture SEPNET-PRISM comparator.

Development-only. This is not the paper's random-IID reported result; it places the
released hybrid sequence/tabular architecture and fixed training recipe onto the
IRIS NEW-crossing chronology. No locked test is accessed.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark_v2 as v2
from iris_report.iris_sep.tools import run_context_stability_diagnostic as cs

EXPECTED_FEATURE_SHA256 = v1.EXPECTED_FEATURE_SHA256
EXPECTED_EVENT_SHA256 = v1.EXPECTED_EVENT_SHA256
REG_TARGETS = ("Future_log_ProtonFlux_max", "Future_log_XRSB_max")
SEEDS = (7, 13, 26, 42, 73)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def finite_or_none(v):
    if isinstance(v, dict): return {str(k): finite_or_none(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [finite_or_none(x) for x in v]
    if isinstance(v, np.ndarray): return [finite_or_none(x) for x in v.tolist()]
    if isinstance(v, np.generic): return finite_or_none(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v

def save_json(path: Path, value):
    path.write_text(json.dumps(finite_or_none(value), indent=2, sort_keys=True, allow_nan=False)+"\n")

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=(0.25, 0.75)):
        super().__init__(); self.gamma=float(gamma); self.alpha=tuple(float(x) for x in alpha)
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        a = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)[targets.long()]
        return (a * (1.0 - pt).pow(self.gamma) * bce).mean()

def shared_trunk(din, dout, p):
    return nn.Sequential(
        nn.LayerNorm(din), nn.GELU(), nn.Dropout(p),
        nn.Linear(din, dout), nn.LayerNorm(dout), nn.GELU()
    )

def cls_mlp(din, hidden, p, n_hidden=2):
    layers=[]; d=din
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p)]
        d=hidden
    layers.append(nn.Linear(d,1))
    return nn.Sequential(*layers)

class PrismHybrid(nn.Module):
    def __init__(self, input_dim: int, reg_dim: int=2):
        super().__init__()
        p=.18
        self.input_ln=nn.LayerNorm(input_dim)
        self.input_drop=nn.Dropout(.10)
        self.lstm=nn.LSTM(input_dim,64,batch_first=True,bidirectional=True)
        self.pos=nn.Sequential(nn.LayerNorm(128),nn.Dropout(p))
        layer=nn.TransformerEncoderLayer(d_model=128,nhead=4,dropout=p,batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=1)
        self.post_ln=nn.LayerNorm(128)
        self.seq_trunk=shared_trunk(128,64,p)
        self.tab_trunk=nn.Sequential(
            nn.Linear(input_dim,80),nn.LayerNorm(80),nn.GELU(),nn.Dropout(p),
            nn.Linear(80,80),nn.LayerNorm(80),nn.GELU(),nn.Dropout(p)
        )
        self.fuse_reg=shared_trunk(144,96,p)
        self.fuse_cls=shared_trunk(144,56,p)
        self.reg_head=nn.Linear(96,reg_dim)
        self.cls_branch=cls_mlp(56,64,p,2)
    def forward(self,x):
        x=self.input_drop(self.input_ln(x))
        last=x[:,-1,:]
        tab=self.tab_trunk(last)
        seq,_=self.lstm(x)
        seq=self.pos(seq)
        seq=self.post_ln(self.transformer(seq))
        seq=self.seq_trunk(seq[:,-1,:])
        fused=torch.cat([seq,tab],dim=1)
        return self.reg_head(self.fuse_reg(fused)), self.cls_branch(self.fuse_cls(fused))

def prepare(features: Path, events: Path):
    if digest(features)!=EXPECTED_FEATURE_SHA256: raise ValueError("feature hash mismatch")
    if digest(events)!=EXPECTED_EVENT_SHA256: raise ValueError("event hash mismatch")
    raw=pd.read_csv(features)
    ev=pd.read_csv(events)
    raw["window_begin"]=pd.to_datetime(raw["window_begin"],utc=True,errors="raise")
    raw["window_end"]=pd.to_datetime(raw["window_end"],utc=True,errors="raise")
    raw=raw.sort_values(["window_end","window_begin"]).reset_index(drop=True)
    y_all, active, event_ids_all, _=v1.derive_target(raw,ev)
    eligible=np.flatnonzero(~active)
    frame=raw.loc[eligible].reset_index(drop=True)
    y=np.asarray(y_all,dtype=np.int8)[eligible]
    event_ids=np.asarray(event_ids_all,dtype=str)[eligible]
    roles, units, purged, positive_units=cs.build_scope_roles(frame,y,event_ids,None)
    fs,_=v2.feature_sets(frame)
    names=fs["FULL_CONTEXT"]
    for c in REG_TARGETS:
        if c not in frame: raise ValueError(f"missing regression target {c}")
    return frame,y,roles,units,purged,positive_units,names

def build_sequences(frame, names, roles):
    fit=roles=="fit"
    if fit.sum()==0: raise ValueError("empty fit")
    scaler=MinMaxScaler()
    scaler.fit(frame.loc[fit,names])
    z=scaler.transform(frame[names])
    z=np.clip(z,0.0,1.0)
    z=np.nan_to_num(z,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)
    times=pd.to_datetime(frame["window_end"],utc=True).astype("int64").to_numpy()
    lag=int(pd.Timedelta(hours=24).value)
    tol=int(pd.Timedelta(hours=3).value)
    sorted_times=np.asarray(times,dtype=np.int64)
    X=np.zeros((len(frame),2,len(names)),dtype=np.float32)
    X[:,1,:]=z
    for i,t in enumerate(sorted_times):
        target=int(t-lag)
        pos=int(np.searchsorted(sorted_times,target))
        best=None; bestd=None
        for j in (pos-1,pos):
            if 0<=j<len(sorted_times) and sorted_times[j] < t:
                d=abs(int(sorted_times[j])-target)
                if d<=tol and (bestd is None or d<bestd):
                    best=j; bestd=d
        if best is not None: X[i,0,:]=z[best]
    reg=frame.loc[:,REG_TARGETS].to_numpy(dtype=np.float64)
    mu=np.nanmean(reg[fit],axis=0); sd=np.nanstd(reg[fit],axis=0)
    sd=np.where((~np.isfinite(sd))|(sd<1e-8),1.0,sd)
    reg=(reg-mu)/sd
    reg=np.nan_to_num(reg,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)
    return X,reg,{"feature_count":len(names),"reg_mu":mu.tolist(),"reg_sd":sd.tolist()}

def best_skill(y,p):
    candidates=np.unique(np.r_[0.01,np.clip(p,0.01,0.99),0.99])
    best=-1e9
    for t in candidates:
        m=v1.threshold_metrics(y,p,float(t))
        score=min(float(m["TSS"]),float(m["HSS"]))
        if score>best: best=score
    return best

@torch.no_grad()
def predict(model,X,batch=512,device="cpu"):
    model.eval(); out=[]
    for i in range(0,len(X),batch):
        xb=torch.from_numpy(X[i:i+batch]).to(device)
        _,logit=model(xb)
        out.append(torch.sigmoid(logit.view(-1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def train_seed(X,reg,y,roles,seed,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1,min(2,torch.get_num_threads())))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

    fit=np.flatnonzero(roles=="fit"); cal=np.flatnonzero(roles=="calibration")
    yfit=y[fit]
    npos=max(1,int(yfit.sum())); nneg=max(1,int(len(yfit)-yfit.sum()))
    sample_w=np.where(yfit==1,0.5/npos,0.5/nneg)
    gen=torch.Generator(); gen.manual_seed(seed)
    sampler=WeightedRandomSampler(torch.tensor(sample_w,dtype=torch.double),len(fit),replacement=True,generator=gen)
    ds=TensorDataset(torch.from_numpy(X[fit]),torch.from_numpy(reg[fit]),torch.from_numpy(yfit.astype(np.float32)))
    loader=DataLoader(ds,batch_size=56,sampler=sampler,num_workers=0)

    model=PrismHybrid(X.shape[-1],reg.shape[1]).to(device)
    reg_names=("fuse_reg","reg_head")
    reg_params=[]; base_params=[]
    for name,p in model.named_parameters():
        (reg_params if any(name.startswith(k) for k in reg_names) else base_params).append(p)
    opt=torch.optim.Adam([
        {"params":base_params,"lr":2.2e-4,"weight_decay":6.5e-5},
        {"params":reg_params,"lr":2.2e-4*2.45,"weight_decay":6.5e-5},
    ])
    prevalence=float(yfit.mean())
    pos_weight=min(4.0,((1-prevalence)/max(prevalence,1e-8))*1.2)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    focal=FocalLoss(1.5,(0.25,0.75))
    regloss=nn.SmoothL1Loss(beta=1.0)

    best=-1e9; best_state=None; patience=0; best_epoch=-1
    history=[]
    for epoch in range(1200):
        if epoch<2: reg_scale=0.0
        elif epoch<10: reg_scale=.80*min(1.0,(epoch-2+1)/8.0)
        else: reg_scale=.80
        model.train()
        for xb,yr,yc in loader:
            xb=xb.to(device); yr=yr.to(device); yc=yc.to(device)
            opt.zero_grad(set_to_none=True)
            rp,z=model(xb); z=z.view(-1)
            lr_reg=regloss(rp,yr)/reg.shape[1]
            lc=bce(z,yc)+.8*focal(z,yc)
            loss=reg_scale*lr_reg+1.45*lc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        pcal=predict(model,X[cal],device=device)
        score=best_skill(y[cal],pcal)
        history.append(float(score))
        if score>best+1e-12:
            best=score; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience=0
        else:
            patience+=1
            if patience>=72: break
    if best_state is None: raise RuntimeError("no checkpoint selected")
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    p=predict(model,X,device=device)
    return p,{
        "seed":int(seed),"best_epoch":int(best_epoch+1),"epochs_run":int(len(history)),
        "best_calibration_min_tss_hss":float(best),"fit_rows":int(len(fit)),
        "fit_positives":int(yfit.sum()),"pos_weight":float(pos_weight),
        "device":str(device)
    }

def run(features:Path,events:Path,output:Path,seed:int):
    if seed not in SEEDS: raise ValueError(f"seed must be one of {SEEDS}")
    output=Path(output)
    if output.exists(): raise ValueError("output must be new")
    output.mkdir(parents=True)
    frame,y,roles,units,purged,positive_units,names=prepare(features,events)
    X,reg,prep=build_sequences(frame,names,roles)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0=time.perf_counter()
    p,train_meta=train_seed(X,reg,y,roles,seed,device)
    np.savez_compressed(output/f"seed_{seed}.npz",
        probability=p,label=y,role=roles,unit_id=units,
        issue_time=frame["window_end"].astype(str).to_numpy())
    receipt={
        "status":"COMPLETED_SEPNET_PRISM_RELEASED_ARCHITECTURE_SEED",
        "scope":"DEVELOPMENT_ONLY_SAME_COHORT_COMPARATOR",
        "seed":seed,"target":v1.TARGET,"locked_test_accessed":False,
        "monitor_rows_accessed_for_evaluation_only":True,
        "feature_table_sha256":digest(features),"event_catalogue_sha256":digest(events),
        "predictor_count":len(names),"preprocessing":prep,"train":train_meta,
        "purged_units":purged,"positive_event_units":int(positive_units),
        "elapsed_sec":float(time.perf_counter()-t0),
        "preregistration":"config/sepnet_prism_architecture_comparator_preregistration_2026-09-05.json"
    }
    save_json(output/f"seed_{seed}.json",receipt)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,required=True); ap.add_argument("--events",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args(); run(a.features,a.events,a.output,a.seed)

if __name__=="__main__": main()
