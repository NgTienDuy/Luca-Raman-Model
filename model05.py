"""model05.py — 1D CNN"""
import numpy as np, time
import torch, torch.nn as nn, torch.optim as optim
from utils import (preprocess_batch, evaluate, print_metrics, save_torch_checkpoint,
                   load_torch_checkpoint, save_results, plot_predictions,
                   plot_predictions_raw, plot_loss_curves, kl_div_loss,
                   get_logger, get_device, INTERRUPTER)
from config import MODEL_CONFIGS, LABEL_COLS, SAVE_EVERY_N_EPOCHS
log=get_logger("model05"); CFG=MODEL_CONFIGS[5]; DEV=get_device()

class CNN1D(nn.Module):
    def __init__(self,in_len,ch,kernels,dropout=0.5,out=6):
        super().__init__()
        enc=[]; prev=1
        for c,k in zip(ch,kernels):
            enc+=[nn.Conv1d(prev,c,k,padding=k//2),nn.BatchNorm1d(c),nn.ReLU(),nn.Dropout(dropout),nn.MaxPool1d(2)]; prev=c
        self.enc=nn.Sequential(*enc); self.gap=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(prev,64),nn.ReLU(),nn.Dropout(0.5),nn.Linear(64,out))
    def forward(self,x): return torch.softmax(self.head(self.gap(self.enc(x.unsqueeze(1)))),dim=-1)

def _aug(X,Y,n=6):
    Xa,Ya=[X],[Y]
    for _ in range(n): Xa.append(X+np.random.randn(*X.shape).astype(np.float32)*0.01); Ya.append(Y)
    for _ in range(n):
        idx=np.random.randint(len(X),size=len(X)); lam=np.random.beta(0.3,0.3,size=(len(X),1)).astype(np.float32)
        Xa.append(lam*X+(1-lam)*X[idx]); Ya.append(lam*Y+(1-lam)*Y[idx])
    return np.vstack(Xa).astype(np.float32),np.vstack(Ya).astype(np.float32)

def run(X_train,X_test,Y_train,Y_test,retrain=False,
        X_test_raw=None,Y_test_raw=None,sid_test_raw=None,**kw):
    log.info("="*60); log.info("Model 5 — 1D CNN")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    Xtr_aug,Ytr_aug=_aug(Xtr,Y_train,n=CFG["n_aug"])
    model=CNN1D(Xtr.shape[1],CFG["channels"],CFG["kernels"],dropout=CFG["dropout"]).to(DEV)
    log.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    opt=optim.AdamW(model.parameters(),lr=CFG["lr"],weight_decay=CFG["weight_decay"])
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2)
    start_ep=0; best_val=np.inf; no_imp=0; hist={"loss":{"train":[],"val":[]}}
    if not retrain:
        ck=load_torch_checkpoint(5,"resume")
        if ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sch.load_state_dict(ck["sch"]); start_ep=ck["epoch"]+1; best_val=ck["best_val"]
            hist=ck["hist"]; no_imp=ck["no_imp"]; log.info(f"  Resume ep {start_ep}")
        else:
            ck2=load_torch_checkpoint(5,"best")
            if ck2:
                model.load_state_dict(ck2["model"]); model.eval()
                with torch.no_grad(): Yp=model(torch.tensor(Xte)).numpy()
                m=evaluate(Y_test,Yp,LABEL_COLS); print_metrics(m,"M5 Test(ck)",2); return Yp,m
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
            opt.zero_grad(); loss=kl_div_loss(model(Xb),Yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); tl+=loss.item(); nb+=1
        sch.step(); tl/=max(nb,1)
        model.eval()
        with torch.no_grad():
            vl=kl_div_loss(model(torch.tensor(Xtr[va_idx]).to(DEV)),torch.tensor(Y_train[va_idx]).to(DEV)).item()
        hist["loss"]["train"].append(tl); hist["loss"]["val"].append(vl)
        if ep%20==0: log.info(f"  Ep{ep:4d}: tr={tl:.5f} val={vl:.5f}")
        if vl<best_val: best_val=vl; no_imp=0; save_torch_checkpoint(5,{"model":model.state_dict()},"best")
        else: no_imp+=1
        if ep%SAVE_EVERY_N_EPOCHS==0:
            save_torch_checkpoint(5,{"model":model.state_dict(),"opt":opt.state_dict(),"sch":sch.state_dict(),"epoch":ep,"best_val":best_val,"hist":hist,"no_imp":no_imp},"resume")
        if no_imp>=CFG["patience"]: log.info(f"  Early stop ep {ep}"); break
    ck2=load_torch_checkpoint(5,"best")
    if ck2: model.load_state_dict(ck2["model"])
    model.eval()
    with torch.no_grad():
        Yp_te=model(torch.tensor(Xte)).numpy()
        Yp_tr=model(torch.tensor(Xtr)).numpy()
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    elapsed=time.time()-t0; log.info(f"  Time: {elapsed:.1f}s")
    print_metrics(mtr,"M5 Train",2); print_metrics(mte,"M5 Test",2)
    plot_loss_curves(hist,5,"1D CNN"); plot_predictions(Y_test,Yp_te,5,"1D CNN",LABEL_COLS)
    if X_test_raw is not None:
        Xte_r=sc.transform(preprocess_batch(X_test_raw,"full")).astype(np.float32)
        with torch.no_grad(): Yp_r=model(torch.tensor(Xte_r)).numpy()
        plot_predictions_raw(Y_test_raw,Yp_r,sid_test_raw,5,"1D CNN",LABEL_COLS)
    save_results(5,mte,{"Y_pred":Yp_te,"Y_true":Y_test},{"name":"1D CNN","elapsed_s":elapsed,"train_metrics":mtr})
    return Yp_te,mte
