"""model07.py — CNN + Bond-Region Attention"""
import numpy as np, time, os
import torch, torch.nn as nn, torch.optim as optim
from utils import (preprocess_batch, evaluate, print_metrics, save_torch_checkpoint,
                   load_torch_checkpoint, save_results, plot_predictions,
                   plot_predictions_raw, plot_loss_curves, plot_bond_detect_batch,
                   kl_div_loss, get_logger, get_device, INTERRUPTER)
from spectral_knowledge import get_active_regions
from config import MODEL_CONFIGS, LABEL_COLS, SAVE_EVERY_N_EPOCHS, get_model_dir
log=get_logger("model07"); CFG=MODEL_CONFIGS[7]; DEV=get_device()

class BondAttnCNN(nn.Module):
    def __init__(self,in_len,ch,kernels,n_bonds,ent_lam,dropout=0.4,out=6):
        super().__init__()
        self.ent_lam=ent_lam; self.n_bonds=n_bonds
        enc=[]; prev=1
        for c,k in zip(ch,kernels):
            enc+=[nn.Conv1d(prev,c,k,padding=k//2),nn.BatchNorm1d(c),nn.ReLU(),nn.Dropout(dropout),nn.MaxPool1d(2)]; prev=c
        self.enc=nn.Sequential(*enc)
        w=torch.ones(9)/9; self.sg_w=nn.Parameter(w); self.pad=4
        self.attn=nn.Sequential(nn.Linear(n_bonds*prev,128),nn.ReLU(),nn.Linear(128,n_bonds))
        self.head=nn.Sequential(nn.Linear(prev,64),nn.ReLU(),nn.Dropout(0.4),nn.Linear(64,out))
        self.ch_last=prev
    def forward(self,x,bond_centers):
        w=torch.softmax(self.sg_w,0).view(1,1,-1)
        xp=nn.functional.pad(x.unsqueeze(1),(self.pad,self.pad),mode="reflect")
        x=nn.functional.conv1d(xp,w).squeeze(1)
        feats=self.enc(x.unsqueeze(1)); B,C,L=feats.shape
        bf=[]; [bf.append(feats[:,:,max(0,min(int(p*L),L-1))]) for p in bond_centers]
        bf=torch.stack(bf,dim=1); attn=torch.softmax(self.attn(bf.reshape(B,-1)),dim=-1)
        ctx=(bf*attn.unsqueeze(-1)).sum(1)
        return torch.softmax(self.head(ctx),dim=-1),attn
    def entropy_loss(self,attn): return -(-(attn*torch.log(attn.clamp(1e-8))).sum(-1)).mean()

def _bond_centers(wns,regions):
    wmin,wmax=float(wns.min()),float(wns.max()); r=wmax-wmin+1e-6
    return [((br.wn_min+br.wn_max)/2-wmin)/r for br in regions]

def _aug(X,Y,n=5):
    Xa,Ya=[X],[Y]
    for _ in range(n): Xa.append(X+np.random.randn(*X.shape).astype(np.float32)*0.01); Ya.append(Y)
    for _ in range(n):
        idx=np.random.randint(len(X),size=len(X)); lam=np.random.beta(0.3,0.3,size=(len(X),1)).astype(np.float32)
        Xa.append(lam*X+(1-lam)*X[idx]); Ya.append(lam*Y+(1-lam)*Y[idx])
    return np.vstack(Xa).astype(np.float32),np.vstack(Ya).astype(np.float32)

def run(X_train,X_test,Y_train,Y_test,wavenumbers=None,retrain=False,
        X_test_raw=None,Y_test_raw=None,sid_test_raw=None,**kw):
    log.info("="*60); log.info("Model 7 — CNN + Bond Attention")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    if wavenumbers is None: wavenumbers=np.linspace(267,2004,X_train.shape[1])
    regions=get_active_regions(wavenumbers); centers=_bond_centers(wavenumbers,regions)
    n_bonds=len(regions)
    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    Xtr_aug,Ytr_aug=_aug(Xtr,Y_train,n=CFG["n_aug"])
    model=BondAttnCNN(Xtr.shape[1],CFG["channels"],CFG["kernels"],n_bonds,CFG["entropy_lambda"],dropout=CFG["dropout"]).to(DEV)
    log.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    opt=optim.AdamW(model.parameters(),lr=CFG["lr"],weight_decay=CFG["weight_decay"])
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2)
    start_ep=0; best_val=np.inf; no_imp=0; hist={"loss":{"train":[],"val":[]}}
    if not retrain:
        ck=load_torch_checkpoint(7,"resume")
        if ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sch.load_state_dict(ck["sch"]); start_ep=ck["epoch"]+1; best_val=ck["best_val"]
            hist=ck["hist"]; no_imp=ck["no_imp"]; log.info(f"  Resume ep {start_ep}")
        else:
            ck2=load_torch_checkpoint(7,"best")
            if ck2:
                model.load_state_dict(ck2["model"]); model.eval()
                with torch.no_grad(): Yp,_=model(torch.tensor(Xte),centers); Yp=Yp.numpy()
                m=evaluate(Y_test,Yp,LABEL_COLS); print_metrics(m,"M7 Test(ck)",2); return Yp,m
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
            opt.zero_grad(); pred,attn=model(Xb,centers)
            loss=kl_div_loss(pred,Yb)+CFG["entropy_lambda"]*model.entropy_loss(attn)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); tl+=loss.item(); nb+=1
        sch.step(); tl/=max(nb,1)
        model.eval()
        with torch.no_grad():
            p,_=model(torch.tensor(Xtr[va_idx]).to(DEV),centers)
            vl=kl_div_loss(p,torch.tensor(Y_train[va_idx]).to(DEV)).item()
        hist["loss"]["train"].append(tl); hist["loss"]["val"].append(vl)
        if ep%20==0: log.info(f"  Ep{ep:4d}: tr={tl:.5f} val={vl:.5f}")
        if vl<best_val: best_val=vl; no_imp=0; save_torch_checkpoint(7,{"model":model.state_dict()},"best")
        else: no_imp+=1
        if ep%SAVE_EVERY_N_EPOCHS==0:
            save_torch_checkpoint(7,{"model":model.state_dict(),"opt":opt.state_dict(),"sch":sch.state_dict(),"epoch":ep,"best_val":best_val,"hist":hist,"no_imp":no_imp},"resume")
        if no_imp>=CFG["patience"]: log.info(f"  Early stop ep {ep}"); break
    ck2=load_torch_checkpoint(7,"best")
    if ck2: model.load_state_dict(ck2["model"])
    model.eval()
    with torch.no_grad():
        Yp_te,attn_te=model(torch.tensor(Xte),centers); Yp_te=Yp_te.numpy(); attn_te=attn_te.numpy()
        Yp_tr,_=model(torch.tensor(Xtr),centers); Yp_tr=Yp_tr.numpy()
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    elapsed=time.time()-t0; log.info(f"  Time: {elapsed:.1f}s")
    print_metrics(mtr,"M7 Train",2); print_metrics(mte,"M7 Test",2)

    # Bond detection for test samples
    if X_test_raw is not None:
        Xte_raw_proc=preprocess_batch(X_test_raw,"full")
        unique_test_sids=list(dict.fromkeys(sid_test_raw))
        for sid in unique_test_sids:
            mask=sid_test_raw==sid; s_rep=np.median(Xte_raw_proc[mask],axis=0)
            idx_s=np.where(mask)[0][0]
            plot_bond_detect_batch(s_rep[None],wavenumbers,[sid],7,"CNN+BondAttn",Y_test_raw[idx_s:idx_s+1],LABEL_COLS)
        Xte_r=sc.transform(Xte_raw_proc).astype(np.float32)
        with torch.no_grad(): Yp_r,_=model(torch.tensor(Xte_r),centers); Yp_r=Yp_r.numpy()
        plot_predictions_raw(Y_test_raw,Yp_r,sid_test_raw,7,"CNN+BondAttn",LABEL_COLS)

    # Attention bar chart
    mean_attn=attn_te.mean(0)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar([br.name for br in regions],mean_attn,color="coral")
    ax.set_title("Bond-Region Attention Weights"); ax.tick_params(axis="x",rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(get_model_dir(7),"model07_bond_attention.png"),dpi=100,bbox_inches="tight"); plt.close()

    plot_loss_curves(hist,7,"CNN+BondAttn"); plot_predictions(Y_test,Yp_te,7,"CNN+BondAttn",LABEL_COLS)
    save_results(7,mte,{"Y_pred":Yp_te,"Y_true":Y_test,"attn":attn_te},
                 {"name":"CNN+Bond Attention","elapsed_s":elapsed,
                  "mean_attention":{br.name:float(w) for br,w in zip(regions,mean_attn)},"train_metrics":mtr})
    try:
        from chemistry_report import batch_chemistry,mean_profile,format_report,plot_bond_contribution
        from dataclasses import asdict; import json
        md=get_model_dir(7); avg=mean_profile(batch_chemistry(Xtr,wavenumbers,Y_train,LABEL_COLS))
        print(format_report(avg,"Model 7 — CNN+Bond Attention",show_composition=True))
        plot_bond_contribution(avg,"M7-CNN-BondAttn",os.path.join(md,"model07_chemistry_bonds.png"))
        with open(os.path.join(md,"model07_chemistry.json"),"w") as ff:
            json.dump({"model":7,"profile":asdict(avg),"mean_attention":{br.name:float(w) for br,w in zip(regions,mean_attn)}},ff,indent=2,default=lambda x:float(x) if hasattr(x,"__float__") else str(x))
    except Exception as ce: log.warning(f"  Chemistry: {ce}")
    return Yp_te,mte
