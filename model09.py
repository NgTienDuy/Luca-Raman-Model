"""model09.py — RL Pipeline Discovery (fixed: accepts wavenumbers kwarg)"""
import numpy as np, time, os
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import r2_score

from utils import (preprocess_batch, evaluate, print_metrics,
                   save_torch_checkpoint, load_torch_checkpoint,
                   save_results, plot_predictions, plot_loss_curves,
                   kl_div_loss, get_logger, get_device, INTERRUPTER,
                   remove_cosmic_rays, savitzky_golay, snip_baseline,
                   als_baseline, snv_normalization, area_normalization,
                   dirichlet_normalize)
from spectral_knowledge import get_active_regions, extract_bond_features
from config import MODEL_CONFIGS, LABEL_COLS, SAVE_EVERY_N_EPOCHS, get_model_dir
log = get_logger("model09"); CFG = MODEL_CONFIGS[9]; DEV = get_device()

TOOLS = {
    0: ("SG",    lambda s: savitzky_golay(s)),
    1: ("SNIP",  lambda s: snip_baseline(s)),
    2: ("ALS",   lambda s: als_baseline(s)),
    3: ("SNV",   lambda s: snv_normalization(s)),
    4: ("Deriv", lambda s: np.gradient(np.gradient(s)).astype(np.float32)),
    5: ("MSC",   lambda s: (s/(np.abs(s).mean()+1e-8)).astype(np.float32)),
    6: ("Cosmic",lambda s: remove_cosmic_rays(s)),
    7: ("Area",  lambda s: area_normalization(s)),
}
N_TOOLS = 8

def _apply(X, tid):
    fn=TOOLS[tid][1]; return np.array([fn(s) for s in X],dtype=np.float32)

class GRUCtrl(nn.Module):
    def __init__(self,enc=64,hid=64,n_tools=8,steps=4):
        super().__init__()
        self.steps=steps; self.hid=hid
        self.enc_net=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Linear(128,enc))
        self.gru=nn.GRUCell(enc+n_tools,hid)
        self.policy=nn.Linear(hid,n_tools)
        self.critic=nn.Linear(hid,1)
    def forward(self,x_enc,h,last_a):
        h=self.gru(torch.cat([x_enc,last_a],dim=-1),h)
        return h,self.policy(h),self.critic(h).squeeze(-1)
    def init_h(self,bs): return torch.zeros(bs,self.hid).to(DEV)

class SEPredictor(nn.Module):
    def __init__(self,in_dim,out=6):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(in_dim,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(0.4),
                                nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3))
        self.se=nn.Sequential(nn.Linear(128,32),nn.ReLU(),nn.Linear(32,128),nn.Sigmoid())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,out))
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.kaiming_normal_(m.weight,nonlinearity="relu"); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward(self,x):
        h=self.enc(x); return torch.softmax(self.head(h*self.se(h)),dim=-1)

def _build_feat(X, wns, nmf=None):
    regions=get_active_regions(wns)
    bf=np.array([extract_bond_features(s,wns,regions) for s in X])
    feats=[X,bf]
    if nmf is not None:
        Xn=np.maximum(X,0); Xn-=Xn.min(axis=1,keepdims=True)
        feats.append(nmf.transform(Xn).astype(np.float32))
    out=np.hstack(feats).astype(np.float32)
    out=np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _augment(X,Y,n=4):
    Xa,Ya=[X],[Y]
    for _ in range(n): Xa.append(X+np.random.randn(*X.shape).astype(np.float32)*0.01); Ya.append(Y)
    return np.vstack(Xa).astype(np.float32),np.vstack(Ya).astype(np.float32)

def run(X_train, X_test, Y_train, Y_test, wavenumbers=None, retrain=False, **kw):
    log.info("="*60); log.info("Model 9 — RL Pipeline Discovery")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    if wavenumbers is None: wavenumbers=np.linspace(400,1800,X_train.shape[1])

    from sklearn.decomposition import NMF
    Xb=np.maximum(preprocess_batch(X_train,"snv"),0); Xb-=Xb.min(axis=1,keepdims=True)
    nmf=NMF(n_components=min(8,len(X_train)-1),init="nndsvda",max_iter=500,random_state=42)
    nmf.fit(Xb); log.info("  NMF done")

    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    from sklearn.preprocessing import StandardScaler
    feat_sc=StandardScaler()
    ftr=feat_sc.fit_transform(_build_feat(Xtr,wavenumbers,nmf)).astype(np.float32)
    fte=feat_sc.transform(_build_feat(Xte,wavenumbers,nmf)).astype(np.float32)
    ftr_aug,Ytr_aug=_augment(ftr,Y_train,n=4)

    in_dim=ftr.shape[1]
    ctrl=GRUCtrl(enc=64,hid=64,n_tools=N_TOOLS,steps=4).to(DEV)
    pred=SEPredictor(in_dim).to(DEV)
    log.info(f"  Predictor params: {sum(p.numel() for p in pred.parameters()):,}")

    opt_p=optim.AdamW(pred.parameters(),lr=CFG["lr"],weight_decay=CFG["weight_decay"])
    opt_c=optim.AdamW(ctrl.parameters(),lr=1e-3)
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_p,T_0=50,T_mult=2)

    start_ep=0; best_val=np.inf; no_imp=0; hist={"loss":{"train":[],"val":[]}}
    tool_usage=np.zeros(N_TOOLS)

    if not retrain:
        ck=load_torch_checkpoint(9,"resume")
        if ck:
            pred.load_state_dict(ck["pred"]); ctrl.load_state_dict(ck["ctrl"])
            opt_p.load_state_dict(ck["opt_p"]); opt_c.load_state_dict(ck["opt_c"])
            start_ep=ck["epoch"]+1; best_val=ck["best_val"]; hist=ck["hist"]
            no_imp=ck["no_imp"]; tool_usage=ck.get("tool_usage",tool_usage)
            log.info(f"  Resume ep {start_ep}")
        else:
            ck2=load_torch_checkpoint(9,"best")
            if ck2:
                pred.load_state_dict(ck2["pred"]); pred.eval()
                with torch.no_grad(): Yp=pred(torch.tensor(fte)).numpy()
                m=evaluate(Y_test,Yp,LABEL_COLS); print_metrics(m,"M9 Test(ck)",2); return Yp,m

    n_val=max(2,int(0.2*len(ftr))); idx=np.random.RandomState(42).permutation(len(ftr)); va_idx=idx[:n_val]
    Xraw_tr=torch.tensor(preprocess_batch(X_train,"snv")[:,:min(1024,X_train.shape[1])],dtype=torch.float32)

    def _iter(X,Y,bs,sh=True):
        N=len(X); perm=np.random.permutation(N) if sh else np.arange(N)
        for i in range(0,N,bs):
            b=perm[i:i+bs]; yield torch.tensor(X[b]).to(DEV),torch.tensor(Y[b]).to(DEV)

    INTERRUPTER.reset(); t0=time.time()
    for ep in range(start_ep,CFG["max_epochs"]):
        if INTERRUPTER.stop_requested: break
        pred.train(); tl=0.; nb=0
        for Xb,Yb in _iter(ftr_aug,Ytr_aug,CFG["batch_size"]):
            opt_p.zero_grad(); loss=kl_div_loss(pred(Xb),Yb)
            loss.backward(); nn.utils.clip_grad_norm_(pred.parameters(),2.); opt_p.step()
            tl+=loss.item(); nb+=1
        sch.step(); tl/=max(nb,1)
        if ep%10==0 and ep>10: _rl_step(ctrl,pred,opt_c,Xraw_tr,ftr,Y_train,tool_usage)
        pred.eval()
        with torch.no_grad():
            vl=kl_div_loss(pred(torch.tensor(ftr[va_idx]).to(DEV)),torch.tensor(Y_train[va_idx]).to(DEV)).item()
        hist["loss"]["train"].append(tl); hist["loss"]["val"].append(vl)
        if ep%20==0: log.info(f"  Ep{ep:4d}: tr={tl:.5f} val={vl:.5f}")
        if vl<best_val: best_val=vl; no_imp=0; save_torch_checkpoint(9,{"pred":pred.state_dict(),"ctrl":ctrl.state_dict()},"best")
        else: no_imp+=1
        if ep%SAVE_EVERY_N_EPOCHS==0:
            save_torch_checkpoint(9,{"pred":pred.state_dict(),"ctrl":ctrl.state_dict(),
                "opt_p":opt_p.state_dict(),"opt_c":opt_c.state_dict(),
                "epoch":ep,"best_val":best_val,"hist":hist,"no_imp":no_imp,"tool_usage":tool_usage},"resume")
        if no_imp>=CFG["patience"]: log.info(f"  Early stop ep {ep}"); break

    ck2=load_torch_checkpoint(9,"best")
    if ck2: pred.load_state_dict(ck2["pred"])
    pred.eval()
    with torch.no_grad():
        Yp_te=pred(torch.tensor(fte)).numpy()
        Yp_tr=pred(torch.tensor(ftr)).numpy()

    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    elapsed=time.time()-t0; log.info(f"  Time: {elapsed:.1f}s")
    print_metrics(mtr,"M9 Train",2); print_metrics(mte,"M9 Test",2)
    tot=tool_usage.sum()+1e-10
    log.info("  Tool usage:"); [log.info(f"    {TOOLS[i][0]:<10}: {tool_usage[i]/tot*100:.1f}%") for i in range(N_TOOLS)]
    plot_loss_curves(hist,9,"RL Pipeline"); plot_predictions(Y_test,Yp_te,9,"RL Pipeline",LABEL_COLS)
    save_results(9,mte,{"Y_pred":Yp_te,"Y_true":Y_test},
                 {"name":"RL Pipeline","elapsed_s":elapsed,
                  "tool_usage":{TOOLS[i][0]:float(tool_usage[i]/tot) for i in range(N_TOOLS)},
                  "train_metrics":mtr})

    # ── Raw scatter ───────────────────────────────────────────────────────
    X_te_raw=kw.get("X_test_raw"); Y_te_raw=kw.get("Y_test_raw"); sid_te_raw=kw.get("sid_test_raw")
    if X_te_raw is not None:
        from utils import plot_predictions_raw, plot_bond_detect_batch
        Xte_r=preprocess_batch(X_te_raw,"full")
        fte_r=feat_sc.transform(np.nan_to_num(_build_feat(Xte_r,wavenumbers,nmf),nan=0.,posinf=0.,neginf=0.)).astype(np.float32)
        with torch.no_grad(): Yp_r=pred(torch.tensor(fte_r)).numpy()
        plot_predictions_raw(Y_te_raw,Yp_r,sid_te_raw,9,"RL Pipeline",LABEL_COLS)
        unique_sids=list(dict.fromkeys(sid_te_raw))
        for sid in unique_sids:
            mask=sid_te_raw==sid; s_rep=np.median(preprocess_batch(X_te_raw[mask],"full"),axis=0); idx_s=np.where(mask)[0][0]
            plot_bond_detect_batch(s_rep[None],wavenumbers,[sid],9,"RL Pipeline",Y_te_raw[idx_s:idx_s+1],LABEL_COLS)

    # ── Chemistry ─────────────────────────────────────────────────────────
    try:
        from chemistry_report import batch_chemistry,mean_profile,format_report,plot_bond_contribution
        import json
        from dataclasses import asdict
        X_proc=preprocess_batch(X_train,"full")
        avg_chem=mean_profile(batch_chemistry(X_proc,wavenumbers,Y_train,LABEL_COLS))
        print(format_report(avg_chem,"Model 9 — RL Pipeline",show_composition=True))
        plot_bond_contribution(avg_chem,"M9-RL",os.path.join(get_model_dir(9),"model09_chemistry_bonds.png"))
        tool_probs={TOOLS[i][0]:float(tool_usage[i]/tot) for i in range(N_TOOLS)}
        with open(os.path.join(get_model_dir(9),"model09_chemistry.json"),"w") as ff:
            json.dump({"model":9,"tool_usage":tool_probs,"profile":asdict(avg_chem)},ff,indent=2,
                      default=lambda x:float(x) if hasattr(x,"__float__") else str(x))
    except Exception as _ce: log.warning(f"  Chemistry lỗi: {_ce}")

    return Yp_te, mte

def _rl_step(ctrl,pred,opt_c,Xraw,ftr,Y,tool_usage):
    bs=min(16,len(Xraw)); idx=np.random.choice(len(Xraw),bs,replace=False)
    Xb=Xraw[idx].to(DEV); Yb=torch.tensor(Y[idx],dtype=torch.float32).to(DEV)
    ctrl.train(); pred.eval()
    h=ctrl.init_h(bs); enc=ctrl.enc_net(Xb); last_a=torch.zeros(bs,N_TOOLS).to(DEV)
    lps=[]; vals=[]; acts=[]
    for _ in range(ctrl.steps):
        h,logits,v=ctrl(enc,h,last_a)
        dist=torch.distributions.Categorical(logits=logits); a=dist.sample()
        lps.append(dist.log_prob(a)); vals.append(v); acts.append(a.cpu().numpy())
        for i in range(N_TOOLS): tool_usage[i]+=float((a==i).sum().cpu())
        last_a=torch.zeros(bs,N_TOOLS).to(DEV); last_a.scatter_(1,a.unsqueeze(1),1.)
    with torch.no_grad():
        try:
            _ptmp = pred(torch.tensor(ftr[idx]).to(DEV)).cpu().numpy()
            _r2   = float(r2_score(Yb.cpu().numpy(), _ptmp, multioutput="uniform_average"))
            if not np.isfinite(_r2): _r2 = 0.
        except Exception:
            _r2 = 0.
    R = torch.tensor([max(_r2, 0.)]*bs, dtype=torch.float32).to(DEV)
    loss=sum(-(lp*(R-v.detach()).clamp(-1,1)).mean()+0.5*((R-v)**2).mean() for lp,v in zip(lps,vals))
    opt_c.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ctrl.parameters(),1.); opt_c.step()

if __name__=="__main__":
    import argparse; from config import DATA_FILE
    from utils import load_data, aggregate_by_sample, split_data
    p=argparse.ArgumentParser(); p.add_argument("--data",default=DATA_FILE); p.add_argument("--retrain",action="store_true"); a=p.parse_args()
    X,Y,s,w=load_data(a.data); Xa,Ya,sa=aggregate_by_sample(X,Y,s)
    Xtr,Xte,Ytr,Yte,_,_=split_data(Xa,Ya,sa); run(Xtr,Xte,Ytr,Yte,w,a.retrain)
