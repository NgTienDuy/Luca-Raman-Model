"""model08.py — Neural Module Network (5 modules)"""
import numpy as np, time, os
import torch, torch.nn as nn, torch.optim as optim

from utils import (preprocess_batch, evaluate, print_metrics,
                   save_torch_checkpoint, load_torch_checkpoint,
                   save_results, plot_predictions, plot_loss_curves,
                   kl_div_loss, get_logger, get_device, INTERRUPTER)
from spectral_knowledge import get_active_regions
from config import MODEL_CONFIGS, LABEL_COLS, SAVE_EVERY_N_EPOCHS, get_model_dir
log = get_logger("model08"); CFG = MODEL_CONFIGS[8]; DEV = get_device()

import torch.nn.functional as F

POLARITY_W   = None  # sẽ khởi tạo khi biết regions
HYDRO_W      = None
PH_W         = None
STRENGTH_W   = None

def _init_chem_weights(regions):
    global POLARITY_W, HYDRO_W, PH_W, STRENGTH_W
    POLARITY_W  = torch.tensor([br.polarity       for br in regions], dtype=torch.float32)
    HYDRO_W     = torch.tensor([br.hydrophilicity for br in regions], dtype=torch.float32)
    PH_W        = torch.tensor([br.ph_effect      for br in regions], dtype=torch.float32)
    STRENGTH_W  = torch.tensor([br.bond_strength  for br in regions], dtype=torch.float32)


class NMN(nn.Module):
    def __init__(self, seq_len, n_filt=32, n_bonds=6, n_unmix=6, out=6):
        super().__init__()
        # M1: peak detection
        self.mu    = nn.Parameter(torch.linspace(0.05,0.95,n_filt))
        self.sigma = nn.Parameter(torch.full((n_filt,),0.08))
        self.register_buffer("pos",torch.linspace(0,1,seq_len))
        # M2: bond assignment
        self.bond_mlp = nn.Sequential(nn.Linear(n_filt,64),nn.ReLU(),nn.Linear(64,n_bonds),nn.Softplus())
        # M3: unmixing
        self.S = nn.Parameter(torch.randn(n_unmix,seq_len)*0.05)
        self.enc_unmix = nn.Sequential(nn.Linear(seq_len,64),nn.ReLU(),nn.Linear(64,n_unmix),nn.Softplus())
        # M5: concentration
        in_dim = n_filt+n_bonds+n_unmix+4
        self.gate=nn.Sequential(nn.Linear(in_dim,128),nn.LayerNorm(128),nn.ReLU(),
                                 nn.Dropout(0.4),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,out))
        self.n_bonds=n_bonds; self.seq_len=seq_len

    def forward(self, x, chem_w):
        pol,hyd,ph,str_w=chem_w
        # M1
        mu=self.mu.unsqueeze(1); sig=torch.abs(self.sigma).unsqueeze(1)+0.02
        kern=torch.exp(-0.5*((self.pos-mu)/sig)**2)
        kern=kern/(kern.sum(1,keepdim=True)+1e-8)
        r=x@kern.T
        # M2
        b=self.bond_mlp(r)
        # M3
        S=torch.nn.functional.softplus(self.S); c=self.enc_unmix(x); x_r=c@S
        # M4: chem props (fixed weights)
        tot=b.sum(-1,keepdim=True)+1e-8; bw=b/tot
        polarity  =(bw*pol.to(DEV)).sum(-1,keepdim=True)
        hydro     =(bw*hyd.to(DEV)).sum(-1,keepdim=True)
        ph_score  =torch.tanh((bw*ph.to(DEV)).sum(-1,keepdim=True))
        strength  =(bw*str_w.to(DEV)).sum(-1,keepdim=True)
        q=torch.cat([polarity,hydro,ph_score,strength],dim=1)
        # M5
        y_hat=torch.softmax(self.gate(torch.cat([r,b,c,q],dim=1)),dim=-1)
        return y_hat, x_r, b, q


def _augment(X,Y,n=5):
    Xa,Ya=[X],[Y]
    for _ in range(n): Xa.append(X+np.random.randn(*X.shape).astype(np.float32)*0.01); Ya.append(Y)
    for _ in range(n):
        idx=np.random.randint(len(X),size=len(X)); lam=np.random.beta(0.3,0.3,size=(len(X),1)).astype(np.float32)
        Xa.append(lam*X+(1-lam)*X[idx]); Ya.append(lam*Y+(1-lam)*Y[idx])
    return np.vstack(Xa).astype(np.float32),np.vstack(Ya).astype(np.float32)


def run(X_train, X_test, Y_train, Y_test, wavenumbers=None, retrain=False, **kw):
    log.info("="*60); log.info("Model 8 — Neural Module Network")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    if wavenumbers is None: wavenumbers=np.linspace(400,1800,X_train.shape[1])
    regions=get_active_regions(wavenumbers); _init_chem_weights(regions)
    chem_w=(POLARITY_W,HYDRO_W,PH_W,STRENGTH_W)
    n_bonds=len(regions)

    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler(); Xtr=sc.fit_transform(Xtr).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    Xtr_aug,Ytr_aug=_augment(Xtr,Y_train,n=5)

    model=NMN(Xtr.shape[1],n_filt=CFG["n_filters"],n_bonds=n_bonds,n_unmix=6).to(DEV)
    log.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    opt=optim.AdamW(model.parameters(),lr=CFG["lr"],weight_decay=CFG["weight_decay"])
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2)

    start_ep=0; best_val=np.inf; no_imp=0; hist={"loss":{"train":[],"val":[]}}
    if not retrain:
        ck=load_torch_checkpoint(8,"resume")
        if ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sch.load_state_dict(ck["sch"]); start_ep=ck["epoch"]+1; best_val=ck["best_val"]
            hist=ck["hist"]; no_imp=ck["no_imp"]; log.info(f"  Resume ep {start_ep}")
        else:
            ck2=load_torch_checkpoint(8,"best")
            if ck2:
                model.load_state_dict(ck2["model"]); model.eval()
                with torch.no_grad(): Yp,_,_,_=model(torch.tensor(Xte).to(DEV),chem_w); Yp=Yp.cpu().numpy()
                m=evaluate(Y_test,Yp,LABEL_COLS); print_metrics(m,"M8 Test(ck)",2); return Yp,m

    n_val=max(2,int(0.2*len(Xtr))); idx=np.random.RandomState(42).permutation(len(Xtr)); va_idx=idx[:n_val]

    def _iter(X,Y,bs,sh=True):
        N=len(X); perm=np.random.permutation(N) if sh else np.arange(N)
        for i in range(0,N,bs):
            b=perm[i:i+bs]; yield torch.tensor(X[b]).to(DEV),torch.tensor(Y[b]).to(DEV)

    INTERRUPTER.reset(); t0=time.time()
    for ep in range(start_ep,CFG["max_epochs"]):
        if INTERRUPTER.stop_requested: break
        model.train(); tl=0.; nb=0
        for Xb,Yb in _iter(Xtr_aug,Ytr_aug,CFG["batch_size"]):
            opt.zero_grad(); pred,xr,bi,q=model(Xb,chem_w)
            loss=kl_div_loss(pred,Yb)+CFG["recon_weight"]*F.mse_loss(xr,Xb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
            tl+=loss.item(); nb+=1
        sch.step(); tl/=max(nb,1)
        model.eval()
        with torch.no_grad():
            p,_,_,_=model(torch.tensor(Xtr[va_idx]).to(DEV),chem_w)
            vl=kl_div_loss(p,torch.tensor(Y_train[va_idx]).to(DEV)).item()
        hist["loss"]["train"].append(tl); hist["loss"]["val"].append(vl)
        if ep%20==0: log.info(f"  Ep{ep:4d}: tr={tl:.5f} val={vl:.5f}")
        if vl<best_val: best_val=vl; no_imp=0; save_torch_checkpoint(8,{"model":model.state_dict()},"best")
        else: no_imp+=1
        if ep%SAVE_EVERY_N_EPOCHS==0:
            save_torch_checkpoint(8,{"model":model.state_dict(),"opt":opt.state_dict(),
                "sch":sch.state_dict(),"epoch":ep,"best_val":best_val,"hist":hist,"no_imp":no_imp},"resume")
        if no_imp>=CFG["patience"]: log.info(f"  Early stop ep {ep}"); break

    ck2=load_torch_checkpoint(8,"best")
    if ck2: model.load_state_dict(ck2["model"])
    model.eval()
    with torch.no_grad():
        Yp_te,_,bi_te,q_te=model(torch.tensor(Xte).to(DEV),chem_w)
        Yp_te=Yp_te.cpu().numpy(); bi_te=bi_te.cpu().numpy(); q_te=q_te.cpu().numpy()
        Yp_tr,_,_,_=model(torch.tensor(Xtr).to(DEV),chem_w); Yp_tr=Yp_tr.cpu().numpy()

    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    elapsed=time.time()-t0; log.info(f"  Time: {elapsed:.1f}s")
    print_metrics(mtr,"M8 Train",2); print_metrics(mte,"M8 Test",2)
    log.info(f"  Chem props [0]: pol={q_te[0,0]:.3f} hyd={q_te[0,1]:.3f} pH={q_te[0,2]:+.3f}")
    plot_loss_curves(hist,8,"NMN"); plot_predictions(Y_test,Yp_te,8,"NMN",LABEL_COLS)
    save_results(8,mte,{"Y_pred":Yp_te,"Y_true":Y_test},
                 {"name":"Neural Module Network","elapsed_s":elapsed,
                  "bond_intensities_s0":{br.name:float(bi_te[0,i]) for i,br in enumerate(regions)},
                  "chem_props_s0":{"polarity":float(q_te[0,0]),"hydrophilicity":float(q_te[0,1]),
                                   "ph_score":float(q_te[0,2]),"bond_strength":float(q_te[0,3])},
                  "train_metrics":mtr})

    # ── Raw scatter ────────────────────────────────────────────────────────
    X_te_raw=kw.get("X_test_raw"); Y_te_raw=kw.get("Y_test_raw"); sid_te_raw=kw.get("sid_test_raw")
    if X_te_raw is not None:
        from utils import plot_predictions_raw, plot_bond_detect_batch
        Xte_r=preprocess_batch(X_te_raw,"full")
        from sklearn.preprocessing import MinMaxScaler as MMS
        sc2=MMS(); sc2.fit(preprocess_batch(X_train,"full"))
        Xte_r2=sc2.transform(Xte_r).astype(np.float32)
        model.eval()
        with torch.no_grad(): Yp_r,_,_,_=model(torch.tensor(Xte_r2).to(DEV),chem_w); Yp_r=Yp_r.cpu().numpy()
        plot_predictions_raw(Y_te_raw,Yp_r,sid_te_raw,8,"NMN",LABEL_COLS)
        unique_sids=list(dict.fromkeys(sid_te_raw))
        for sid in unique_sids:
            mask=sid_te_raw==sid; s_rep=np.median(Xte_r[mask],axis=0); idx_s=np.where(mask)[0][0]
            plot_bond_detect_batch(s_rep[None],wavenumbers,[sid],8,"NMN",Y_te_raw[idx_s:idx_s+1],LABEL_COLS)

    # ── Chemistry from NMN module outputs ─────────────────────────────────
    try:
        from chemistry_report import (extract_chemistry_profile, batch_chemistry,
                                       mean_profile, format_report, plot_bond_contribution,
                                       ChemistryProfile, save_chemistry_json)
        from dataclasses import asdict; import json

        X_proc=preprocess_batch(X_train,"full")
        from sklearn.preprocessing import MinMaxScaler as MMS2
        sc3=MMS2(); X_p2=sc3.fit_transform(X_proc).astype(np.float32)
        model.eval()
        all_bi=[]; all_q=[]
        bs2=16
        for s in range(0,len(X_p2),bs2):
            xb=torch.tensor(X_p2[s:s+bs2]).to(DEV)
            with torch.no_grad(): _,_,bi_b,q_b=model(xb,chem_w)
            all_bi.append(bi_b.cpu().numpy()); all_q.append(q_b.cpu().numpy())
        all_bi=np.vstack(all_bi); all_q=np.vstack(all_q)
        # Build ChemistryProfile from NMN outputs
        nmn_profiles=[]
        for i in range(len(X_p2)):
            prof=extract_chemistry_profile(X_proc[i],wavenumbers,Y_train[i],LABEL_COLS)
            # Override with NMN-specific values
            prof.polarity      = float(all_q[i,0])
            prof.hydrophilicity= float(all_q[i,1])
            prof.pH_score      = float(all_q[i,2])
            prof.bond_strength = float(all_q[i,3])
            # Bond intensities from M2
            for k,br in enumerate(regions):
                if k < all_bi.shape[1]:
                    prof.bond_intensities[br.name]=float(all_bi[i,k])
            nmn_profiles.append(prof)
        avg_nmn=mean_profile(nmn_profiles)
        print(format_report(avg_nmn,"Model 8 — Neural Module Network",show_composition=True))
        plot_bond_contribution(avg_nmn,"M8-NMN",os.path.join(get_model_dir(8),"model08_chemistry_bonds.png"))
        with open(os.path.join(get_model_dir(8),"model08_chemistry.json"),"w") as ff:
            json.dump({"model":8,"nmn_chem_props":{"polarity":float(all_q[:,0].mean()),
                "hydrophilicity":float(all_q[:,1].mean()),"ph_score":float(all_q[:,2].mean()),
                "bond_strength":float(all_q[:,3].mean())},"profile":asdict(avg_nmn)},ff,indent=2,
                default=lambda x:float(x) if hasattr(x,"__float__") else str(x))
        log.info("  ✓ NMN chemistry report saved")
    except Exception as _ce: log.warning(f"  Chemistry lỗi: {_ce}")

    return Yp_te, mte

if __name__=="__main__":
    import argparse; from config import DATA_FILE
    from utils import load_data, aggregate_by_sample, split_data
    p=argparse.ArgumentParser(); p.add_argument("--data",default=DATA_FILE); p.add_argument("--retrain",action="store_true"); a=p.parse_args()
    X,Y,s,w=load_data(a.data); Xa,Ya,sa=aggregate_by_sample(X,Y,s)
    Xtr,Xte,Ytr,Yte,_,_=split_data(Xa,Ya,sa); run(Xtr,Xte,Ytr,Yte,w,a.retrain)
