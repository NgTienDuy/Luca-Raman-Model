"""model10.py — Radial Information Expansion Representation (RIER)"""
import numpy as np, time, os
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F

from utils import (preprocess_batch, evaluate, print_metrics,
                   save_torch_checkpoint, load_torch_checkpoint, save_results,
                   plot_predictions, plot_loss_curves, kl_div_loss,
                   get_logger, get_device, INTERRUPTER, information_score, mutual_info_score)
from spectral_knowledge import (get_active_regions, extract_bond_features, extract_derivative_features)
from config import MODEL_CONFIGS, LABEL_COLS, get_model_dir, SAVE_EVERY_N_EPOCHS
log=get_logger("model10"); CFG=MODEL_CONFIGS[10]; DEV=get_device()
SPOKE_NAMES=["Raw-AE","PCA","FFT","Bond","Deriv","NMF","Moments","XCorr"]

class SpokeAE(nn.Module):
    def __init__(self,d,lat):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,lat))
        self.dec=nn.Sequential(nn.Linear(lat,128),nn.ReLU(),nn.Linear(128,d))
    def forward(self,x): z=self.enc(x); return z,self.dec(z)

class SpokePCA(nn.Module):
    def __init__(self,d,lat):
        super().__init__(); self.W=nn.Linear(d,lat,bias=False); nn.init.orthogonal_(self.W.weight)
    def forward(self,x): return self.W(x),None

class SpokeFFT(nn.Module):
    def __init__(self,n_fft,lat):
        super().__init__(); self.n=n_fft; self.net=nn.Sequential(nn.Linear(n_fft,lat),nn.ReLU())
    def forward(self,x): return self.net(torch.fft.rfft(x,norm="ortho").abs()[:,:self.n]),None

class SpokeMLP(nn.Module):
    def __init__(self,d,lat):
        super().__init__(); h=max(d//2,lat)
        self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,lat))
    def forward(self,x): return self.net(x),None

class GatingFusion(nn.Module):
    def __init__(self,lat,n):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(lat*n,256),nn.ReLU(),nn.Linear(256,n),nn.Sigmoid())
        self.norm=nn.LayerNorm(lat)
    def forward(self,zs):
        g=self.gate(torch.cat(zs,dim=-1))
        fused=(torch.stack(zs,dim=1)*g.unsqueeze(-1)).sum(1)
        return self.norm(fused),g

class RIER(nn.Module):
    def __init__(self,sd,bd,dd,nd,md,xd,lat=64,nf=32,out=6):
        super().__init__()
        self.s0=SpokeAE(sd,lat); self.s1=SpokePCA(sd,lat)
        self.s2=SpokeFFT(min(nf,sd//2+1),lat); self.s3=SpokeMLP(bd,lat)
        self.s4=SpokeMLP(dd,lat); self.s5=SpokeMLP(nd,lat)
        self.s6=SpokeMLP(md,lat); self.s7=SpokeMLP(xd,lat)
        self.fuse=GatingFusion(lat,8)
        self.head=nn.Sequential(nn.Linear(lat,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,out))
        for m in self.modules():
            if isinstance(m,nn.Linear) and m is not self.s1.W:
                nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,b):
        z0,xr=self.s0(b["spec"]); z1,_=self.s1(b["spec"]); z2,_=self.s2(b["spec"])
        z3,_=self.s3(b["bond"]); z4,_=self.s4(b["deriv"]); z5,_=self.s5(b["nmf"])
        z6,_=self.s6(b["mom"]); z7,_=self.s7(b["xcorr"])
        zs=[z0,z1,z2,z3,z4,z5,z6,z7]; fused,g=self.fuse(zs)
        return torch.softmax(self.head(fused),dim=-1),g,xr,zs

def _moments(X,n=8):
    from scipy import stats
    N=len(X); segs=np.array_split(np.arange(X.shape[1]),n); out=np.zeros((N,4*n),dtype=np.float32)
    for i,s in enumerate(X):
        c=0
        for seg in segs:
            sg=s[seg]; out[i,c]=sg.mean(); out[i,c+1]=sg.std()
            out[i,c+2]=float(stats.skew(sg)); out[i,c+3]=float(stats.kurtosis(sg)); c+=4
    return out

def _xcorr(X,n=16):
    N,L=X.shape; h=L//2; out=np.zeros((N,n),dtype=np.float32)
    for i,s in enumerate(X):
        a=s[:h]-s[:h].mean(); b=s[h:h+h]-s[h:h+h].mean(); ml=min(len(a),len(b))
        corr=np.correlate(a[:ml],b[:ml],"full"); mid=len(corr)//2; hn=n//2
        out[i]=corr[mid-hn:mid+hn]
    return (out/(np.abs(out).max(1,keepdims=True)+1e-8)).astype(np.float32)

def _prep(X,wns,nmf):
    from sklearn.preprocessing import StandardScaler
    reg=get_active_regions(wns)
    bf=np.array([extract_bond_features(s,wns,reg) for s in X])
    df=np.array([extract_derivative_features(s,wns,n_seg=6) for s in X])
    mom=_moments(X); xcr=_xcorr(X)
    Xn=np.maximum(X,0); Xn-=Xn.min(axis=1,keepdims=True)
    nf=nmf.transform(Xn).astype(np.float32)
    return {"spec":X.astype(np.float32),"bond":bf,"deriv":df,"nmf":nf,"mom":mom,"xcorr":xcr}

def _to_dev(d): return {k:torch.tensor(v).to(DEV) for k,v in d.items()}
def _sl(d,idx): return {k:v[idx] for k,v in d.items()}

def _aug(d,Y,n=4):
    aug={k:[v] for k,v in d.items()}; Ya=[Y]; N=len(Y)
    for _ in range(n):
        for k in aug: aug[k].append(d[k]+np.random.randn(*d[k].shape).astype(np.float32)*0.01)
        Ya.append(Y)
    for _ in range(n):
        idx=np.random.randint(N,size=N); lam=np.random.beta(0.3,0.3,size=(N,1)).astype(np.float32)
        for k in aug: aug[k].append(lam*d[k]+(1-lam)*d[k][idx])
        Ya.append(lam*Y+(1-lam)*Y[idx])
    return {k:np.vstack(v).astype(np.float32) for k,v in aug.items()},np.vstack(Ya).astype(np.float32)

def _pred_all(model,feats,bs):
    N=len(feats["spec"]); ps=[]
    for s in range(0,N,bs):
        with torch.no_grad(): p,_,_,_=model(_to_dev(_sl(feats,np.arange(s,min(s+bs,N))))); ps.append(p.cpu().numpy())
    return np.vstack(ps)

def _gates_all(model,feats,bs):
    N=len(feats["spec"]); gs=[]
    for s in range(0,N,bs):
        with torch.no_grad(): _,g,_,_=model(_to_dev(_sl(feats,np.arange(s,min(s+bs,N))))); gs.append(g.cpu().numpy())
    return np.vstack(gs)

def _latents_all(model,feats,bs):
    N=len(feats["spec"]); lats=[[] for _ in range(8)]
    for s in range(0,N,bs):
        with torch.no_grad(): _,_,_,zs=model(_to_dev(_sl(feats,np.arange(s,min(s+bs,N)))))
        for k,z in enumerate(zs): lats[k].append(z.cpu().numpy())
    return [np.vstack(l) for l in lats]

def _info(lats,gates,Y):
    rep={}
    for k,(nm,z) in enumerate(zip(SPOKE_NAMES,lats)):
        er=information_score(z); mi=mutual_info_score(z,Y); gw=float(gates[:,k].mean())
        rep[nm]={"effective_rank":er,"mi_score":mi,"gate_weight":gw,"radial_score":er/100*0.4+mi*0.4+gw*0.2}
    rep["total_radius"]=sum(v["radial_score"] for v in rep.values() if isinstance(v,dict))
    return rep

def _print_info(rep):
    log.info(f"  {'Spoke':<12} {'Eff.Rank':>10} {'MI':>8} {'Gate':>8} {'Radius':>10}")
    log.info("  "+"─"*52)
    for n,v in sorted([(k,v) for k,v in rep.items() if isinstance(v,dict)],key=lambda x:x[1]["radial_score"],reverse=True):
        log.info(f"  {n:<12} {v['effective_rank']:>10.2f} {v['mi_score']:>8.4f} {v['gate_weight']:>8.4f} {v['radial_score']:>10.4f}")
    log.info(f"  Total radius = {rep.get('total_radius',0):.4f}")

def _plot_radial(rep, mid):
    """Spoke flowgraph: 8 nhánh xử lý hội tụ về dự đoán cuối cùng"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    spk = {k:v for k,v in rep.items() if isinstance(v,dict)}
    # Định nghĩa pipeline cho mỗi spoke (input → method → output)
    PIPELINES = {
        "Raw-AE":  ("Raw spectrum",  ["Autoencoder"],         "Latent-AE",   "#E8593C"),
        "PCA":     ("Raw spectrum",  ["PCA"],                  "PCA scores",  "#3B8BD4"),
        "FFT":     ("Raw spectrum",  ["FFT", "Filter"],        "Freq. repr.", "#534AB7"),
        "Bond":    ("Preprocessed",  ["ALS","Bond extract"],   "Bond feats",  "#1D9E75"),
        "Deriv":   ("Preprocessed",  ["SG deriv."],            "Deriv. feats","#EF9F27"),
        "NMF":     ("Preprocessed",  ["NMF (MCR-ALS)"],        "Comp. conc.", "#D4537E"),
        "Moments": ("Preprocessed",  ["Seg. moments"],         "Stat. feats", "#5DCAA5"),
        "XCorr":   ("Raw spectrum",  ["Cross-corr."],          "Corr. feats", "#888780"),
    }
    n_spokes = len(PIPELINES)
    fig_h = n_spokes * 1.2 + 2.0
    fig, ax = plt.subplots(figsize=(14, fig_h)); ax.set_xlim(0,14); ax.set_ylim(0, fig_h)
    ax.axis("off"); ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    # Header
    ax.text(7, fig_h-0.6, "RIER — Spoke Processing Flowgraph",
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(7, fig_h-1.1, "Input → Methods → Feature → Fusion → Prediction",
            ha="center", va="center", fontsize=9, color="#555")

    # Column x-positions
    COL = {"input":1.2, "methods":4.5, "feature":9.0, "gate":11.5, "fusion":13.2}
    # Draw column labels
    for label, x in [("Input source",1.2),("Processing pipeline",4.5),("Feature output",9.0),("Gate",11.5)]:
        ax.text(x, fig_h-1.6, label, ha="center", fontsize=8, color="#777", style="italic")

    # Fusion box (right side, center)
    fy = fig_h/2 - 0.3
    fusion_box = FancyBboxPatch((12.5, fy-0.5), 1.2, 1.0, boxstyle="round,pad=0.1",
                                 fc="#2C2C2A", ec="white", lw=1.5)
    ax.add_patch(fusion_box)
    ax.text(13.1, fy, "Gated\nFusion", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Prediction node
    pred_box = FancyBboxPatch((12.6, fy-2.0), 1.0, 0.7, boxstyle="round,pad=0.1",
                               fc="#534AB7", ec="white", lw=1.5)
    ax.add_patch(pred_box)
    ax.text(13.1, fy-1.65, "Pred\n(AA%)", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.annotate("", xy=(13.1, fy-1.3), xytext=(13.1, fy-0.5),
                arrowprops=dict(arrowstyle="->", color="#534AB7", lw=2))

    spoke_names = list(PIPELINES.keys())
    y_positions = np.linspace(fig_h-2.3, 0.7, n_spokes)

    for i, (spoke_name, y_pos) in enumerate(zip(spoke_names, y_positions)):
        src, methods, output, color = PIPELINES[spoke_name]
        gate_w = spk.get(spoke_name, {}).get("gate_weight", 0.)
        mi_v   = spk.get(spoke_name, {}).get("mi_score", 0.)
        r_v    = spk.get(spoke_name, {}).get("radial_score", 0.)

        alpha = 0.5 + 0.5 * min(gate_w * 3, 1.)  # brighter = higher gate

        # ── Input bubble ──────────────────────────────────────────────────
        ax.text(COL["input"], y_pos, src, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="white", alpha=0.25), color="#333")

        # ── Arrow input→methods ───────────────────────────────────────────
        ax.annotate("", xy=(COL["methods"]-1.3, y_pos), xytext=(COL["input"]+0.55, y_pos),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=alpha))

        # ── Method nodes (chain) ──────────────────────────────────────────
        n_m = len(methods)
        x_starts = np.linspace(COL["methods"]-1.2, COL["methods"]+1.2, n_m)
        for j, (method, xm) in enumerate(zip(methods, x_starts)):
            mb = FancyBboxPatch((xm-0.55, y_pos-0.22), 1.1, 0.44,
                                boxstyle="round,pad=0.08", fc=color, ec="white",
                                alpha=alpha*0.8, lw=1.)
            ax.add_patch(mb)
            ax.text(xm, y_pos, method, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
            if j < n_m-1:
                ax.annotate("", xy=(x_starts[j+1]-0.55, y_pos),
                            xytext=(xm+0.55, y_pos),
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.7))

        # ── Arrow methods→feature ─────────────────────────────────────────
        ax.annotate("", xy=(COL["feature"]-0.7, y_pos), xytext=(x_starts[-1]+0.55, y_pos),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=alpha))

        # ── Feature output box ────────────────────────────────────────────
        fb = FancyBboxPatch((COL["feature"]-0.65, y_pos-0.22), 1.3, 0.44,
                            boxstyle="round,pad=0.08", fc=color, ec="white", alpha=0.2, lw=1.)
        ax.add_patch(fb)
        ax.text(COL["feature"], y_pos, output, ha="center", va="center",
                fontsize=7.5, color="#333", style="italic")

        # ── Gate bar ──────────────────────────────────────────────────────
        bar_w = gate_w * 1.5; bar_x = COL["gate"]-0.75
        ax.barh(y_pos, bar_w, left=bar_x, height=0.28, color=color, alpha=0.7)
        ax.text(bar_x + max(bar_w, 0.02) + 0.05, y_pos, f"{gate_w:.3f}",
                va="center", fontsize=7, color="#333")

        # ── Arrow feature→fusion (curved) ─────────────────────────────────
        ax.annotate("",
                    xy=(12.5, fy),
                    xytext=(COL["feature"]+0.65, y_pos),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.3,
                                    connectionstyle=f"arc3,rad={0.1*(i-n_spokes//2)*0.15}"))

        # ── Spoke label (right margin) ────────────────────────────────────
        ax.text(0.1, y_pos, f"{spoke_name}", ha="left", va="center",
                fontsize=8, color=color, fontweight="bold")

    # ── Gate column header bar ────────────────────────────────────────────────
    ax.axvline(x=COL["gate"]-0.75, color="#ccc", lw=0.5, ymin=0.05, ymax=0.93)
    ax.text(COL["gate"]+0.35, fig_h-1.6, "Gate\n(0→1)", ha="center", fontsize=7, color="#777", style="italic")

    # ── Info box ──────────────────────────────────────────────────────────────
    info_str = "\n".join([
        f"nhánh {i+1}: {nm} → {PIPELINES[nm][1][0]} → {PIPELINES[nm][2]}"
        for i, nm in enumerate(spoke_names)
    ])
    ax.text(0.1, 0.3, info_str, ha="left", va="bottom", fontsize=6.5,
            color="#555", family="monospace")

    plt.tight_layout(rect=[0,0,1,1])
    out_path = os.path.join(get_model_dir(mid), f"model{mid:02d}_radial_info.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

def run(X_train, X_test, Y_train, Y_test, wavenumbers=None, retrain=False, **kw):
    log.info("="*60); log.info("Model 10 — RIER"); log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    if wavenumbers is None: wavenumbers=np.linspace(400,1800,X_train.shape[1])
    from sklearn.decomposition import NMF; from sklearn.preprocessing import MinMaxScaler, StandardScaler
    Xb=np.maximum(preprocess_batch(X_train,"snv"),0); Xb-=Xb.min(axis=1,keepdims=True)
    nmf=NMF(n_components=min(8,len(X_train)-1),init="nndsvda",max_iter=500,random_state=42); nmf.fit(Xb)
    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    sc=MinMaxScaler(); Xtr=sc.fit_transform(Xtr).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    log.info("  Trích xuất 8 spoke features...")
    ftr=_prep(Xtr,wavenumbers,nmf); fte=_prep(Xte,wavenumbers,nmf)
    for k in ["bond","deriv","nmf","mom","xcorr"]:
        s2=StandardScaler(); ftr[k]=s2.fit_transform(ftr[k]).astype(np.float32); fte[k]=s2.transform(fte[k]).astype(np.float32)
    faug,Yaug=_aug(ftr,Y_train,n=4); dims={k:ftr[k].shape[1] for k in ftr}
    log.info(f"  Dims: {dims}")
    model=RIER(dims["spec"],dims["bond"],dims["deriv"],dims["nmf"],dims["mom"],dims["xcorr"],
               lat=CFG["latent_dim"],nf=min(CFG["n_fft_components"],dims["spec"]//2+1)).to(DEV)
    log.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    opt=optim.AdamW(model.parameters(),lr=CFG["lr"],weight_decay=CFG["weight_decay"])
    sch=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=2)
    start_ep=0; best_val=-np.inf; no_imp=0; hist={"R2":{"train":[],"val":[]}}
    if not retrain:
        ck=load_torch_checkpoint(10,"resume")
        if ck:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"])
            start_ep=ck["epoch"]+1; best_val=ck["best_val"]; hist=ck["hist"]; no_imp=ck["no_imp"]
            log.info(f"  Resume ep {start_ep}")
        else:
            ck2=load_torch_checkpoint(10,"best")
            if ck2:
                model.load_state_dict(ck2["model"]); model.eval()
                Yp=_pred_all(model,fte,CFG["batch_size"]); m=evaluate(Y_test,Yp,LABEL_COLS)
                print_metrics(m,"M10 Test(ck)",2)
                info=_info(_latents_all(model,fte,CFG["batch_size"]),_gates_all(model,fte,CFG["batch_size"]),Y_test)
                _print_info(info); _plot_radial(info,10); return Yp,m
    from sklearn.metrics import r2_score as r2
    n_val=max(2,int(0.2*len(ftr["spec"]))); idx=np.random.RandomState(42).permutation(len(ftr["spec"])); va_idx=idx[:n_val]
    fva=_sl(ftr,va_idx); N=len(faug["spec"]); bs=CFG["batch_size"]
    INTERRUPTER.reset(); t0=time.time()
    for ep in range(start_ep,CFG["max_epochs"]):
        if INTERRUPTER.stop_requested: break
        model.train(); tl=0.; nb=0; perm=np.random.permutation(N)
        for s in range(0,N,bs):
            b=perm[s:s+bs]; batch=_to_dev(_sl(faug,b)); Yb=torch.tensor(Yaug[b],dtype=torch.float32).to(DEV)
            opt.zero_grad(); p,g,xr,zs=model(batch)
            loss=kl_div_loss(p,Yb)+CFG["recon_weight"]*F.mse_loss(xr,batch["spec"])
            div=sum(F.cosine_similarity(zs[i],zs[j]).clamp(0).mean() for i in range(8) for j in range(i+1,8))/28
            (loss+CFG["diversity_weight"]*div).backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); tl+=loss.item(); nb+=1
        sch.step(); tl/=max(nb,1)
        model.eval()
        with torch.no_grad():
            pv,_,_,_=model(_to_dev(fva)); vl=r2(Y_train[va_idx],pv.cpu().numpy(),multioutput="uniform_average")
        hist["R2"]["train"].append(-tl); hist["R2"]["val"].append(vl)
        if ep%20==0: log.info(f"  Ep{ep:4d}: loss={tl:.5f} val_R²={vl:.4f}")
        if vl>best_val: best_val=vl; no_imp=0; save_torch_checkpoint(10,{"model":model.state_dict()},"best")
        else: no_imp+=1
        if ep%SAVE_EVERY_N_EPOCHS==0:
            save_torch_checkpoint(10,{"model":model.state_dict(),"opt":opt.state_dict(),"sch":sch.state_dict(),"epoch":ep,"best_val":best_val,"hist":hist,"no_imp":no_imp},"resume")
        if no_imp>=CFG["patience"]: log.info(f"  Early stop ep {ep}"); break
    ck2=load_torch_checkpoint(10,"best")
    if ck2: model.load_state_dict(ck2["model"])
    model.eval()
    Yp_te=_pred_all(model,fte,bs); Yp_tr=_pred_all(model,ftr,bs)
    lats_te=_latents_all(model,fte,bs); gates_te=_gates_all(model,fte,bs)
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    elapsed=time.time()-t0; log.info(f"  Time: {elapsed:.1f}s")
    print_metrics(mtr,"M10 Train",2); print_metrics(mte,"M10 Test",2)
    log.info("\n  ─── Radial Information Report ───────────────")
    info=_info(lats_te,gates_te,Y_test); _print_info(info); _plot_radial(info,10)
    plot_loss_curves(hist,10,"RIER"); plot_predictions(Y_test,Yp_te,10,"RIER",LABEL_COLS)

    # ── Raw scatter (QUAN TRỌNG: 6 nhóm điểm) ─────────────────────────────
    X_te_raw=kw.get("X_test_raw"); Y_te_raw=kw.get("Y_test_raw"); sid_te_raw=kw.get("sid_test_raw")
    if X_te_raw is not None:
        from utils import plot_predictions_raw
        Xte_r=preprocess_batch(X_te_raw,"full")
        sc_r=MinMaxScaler(); sc_r.fit(preprocess_batch(X_train,"full"))
        Xte_r2=sc_r.transform(Xte_r).astype(np.float32)
        fte_r=_prep(Xte_r2,wavenumbers,nmf)
        for k2 in ["bond","deriv","nmf","mom","xcorr"]:
            s_tmp=StandardScaler(); s_tmp.fit(ftr[k2]); fte_r[k2]=s_tmp.transform(fte_r[k2]).astype(np.float32)
        Yp_r=_pred_all(model,fte_r,bs)
        plot_predictions_raw(Y_te_raw,Yp_r,sid_te_raw,10,"RIER",LABEL_COLS)
        from utils import plot_bond_detect_batch
        unique_sids=list(dict.fromkeys(sid_te_raw))
        for sid in unique_sids:
            mask=sid_te_raw==sid; s_rep=np.median(Xte_r[mask],axis=0); idx_s=np.where(mask)[0][0]
            plot_bond_detect_batch(s_rep[None],wavenumbers,[sid],10,"RIER",Y_te_raw[idx_s:idx_s+1],LABEL_COLS)

    # ── COMPREHENSIVE CHEMISTRY — phần chính của RIER ─────────────────────
    try:
        from chemistry_report import (batch_chemistry, mean_profile, format_report,
                                       plot_bond_contribution, compare_models,
                                       save_chemistry_json, ChemistryProfile)
        from dataclasses import asdict; import json


        X_proc=preprocess_batch(X_train,"full")
        train_profiles=batch_chemistry(X_proc,wavenumbers,Y_train,LABEL_COLS)
        avg_chem=mean_profile(train_profiles)

        # ── 1. Báo cáo tổng hợp ───────────────────────────────────────────
        print(); print("="*60)
        print("  RIER — PHÂN TÍCH HÓA-LÝ TOÀN DIỆN")
        print("="*60)
        print(format_report(avg_chem,"Model 10 — RIER (train samples avg)",show_composition=True))

        # ── 2. Per-sample chemistry ───────────────────────────────────────
        print(); print("  Per-sample physicochemical properties:")
        print(f"  {'Sample':<8} {'pI':>6} {'pH':>7} {'Polar':>7} {'Hydro':>7} {'Arom':>7} {'Struct'}")
        print(f"  {'─'*58}")
        for i, (prof, sid_i) in enumerate(zip(train_profiles, range(len(train_profiles)))):
            pi_est=prof.estimated_pI if prof.estimated_pI>0 else 0.
            print(f"  {str(sid_i):<8} {pi_est:>6.2f} {prof.pH_score:>+7.3f} "
                  f"{prof.polarity:>7.3f} {prof.hydrophilicity:>7.3f} "
                  f"{prof.aromaticity:>7.4f} {prof.secondary_structure[:12]}")

        # ── 3. Bond contribution plot ─────────────────────────────────────
        plot_bond_contribution(avg_chem,"M10-RIER",
                               os.path.join(get_model_dir(10),"model10_chemistry_bonds.png"))

        # ── 4. Spoke × Chemistry correlation ──────────────────────────────
        spoke_chem = {}
        for k, (sname, z) in enumerate(zip(SPOKE_NAMES, lats_te)):
            # Mutual info của spoke latent với chemistry properties
            from utils import mutual_info_score
            chem_targets = np.column_stack([
                [p.pH_score for p in train_profiles],
                [p.polarity for p in train_profiles],
                [p.hydrophilicity for p in train_profiles],
                [p.aromaticity for p in train_profiles],
            ])[:len(z)]
            if len(chem_targets) > 1 and len(z) > 1:
                mi = mutual_info_score(z[:len(chem_targets)], chem_targets)
            else: mi = 0.
            spoke_chem[sname] = {"latent_chem_mi": float(mi),
                                  "gate_weight": float(gates_te[:,k].mean())}

        print("  Spoke → Chemistry MI:")
        for sn, sv in sorted(spoke_chem.items(), key=lambda x:x[1]["latent_chem_mi"], reverse=True):
            print(f"    {sn:<12}: MI={sv['latent_chem_mi']:.4f}  Gate={sv['gate_weight']:.4f}")

        # ── 5. Save everything ────────────────────────────────────────────
        out = {
            "model": 10, "n_spokes": 8,
            "avg_profile": asdict(avg_chem),
            "spoke_chemistry_mi": spoke_chem,
            "information_report": info,
            "per_sample": [
                {"estimated_pI":p.estimated_pI,"pH_score":p.pH_score,
                 "polarity":p.polarity,"hydrophilicity":p.hydrophilicity,
                 "aromaticity":p.aromaticity,"bond_strength":p.bond_strength,
                 "aliphatic_index":p.aliphatic_index,"carboxylate_ratio":p.carboxylate_ratio,
                 "crystallinity_proxy":p.crystallinity_proxy,"spectral_entropy":p.spectral_entropy,
                 "secondary_structure":p.secondary_structure,"dominant_bond":p.dominant_bond}
                for p in train_profiles
            ]
        }
        with open(os.path.join(get_model_dir(10),"model10_chemistry.json"),"w") as ff:
            json.dump(out,ff,indent=2,
                      default=lambda x:float(x) if hasattr(x,"__float__") else str(x))
        log.info("  ✓ RIER chemistry analysis saved to model10_chemistry.json")

    except Exception as _ce:
        import traceback as _tb
        log.warning(f"  Chemistry lỗi: {_ce}"); log.warning(_tb.format_exc())

    save_results(10,mte,{"Y_pred":Yp_te,"Y_true":Y_test,"gates":gates_te},
                 {"name":"RIER","elapsed_s":elapsed,
                  "information_report":{k:{sk:float(sv) for sk,sv in v.items()} if isinstance(v,dict) else float(v) for k,v in info.items()},
                  "train_metrics":mtr})
    return Yp_te,mte

if __name__=="__main__":
    import argparse; from config import DATA_FILE
    from utils import load_data, aggregate_by_sample, split_data
    p=argparse.ArgumentParser(); p.add_argument("--data",default=DATA_FILE); p.add_argument("--retrain",action="store_true"); a=p.parse_args()
    X,Y,s,w=load_data(a.data); Xa,Ya,sa=aggregate_by_sample(X,Y,s)
    Xtr,Xte,Ytr,Yte,_,_=split_data(Xa,Ya,sa); run(Xtr,Xte,Ytr,Yte,w,a.retrain)
