"""
utils.py — Tiện ích: dữ liệu, tiền xử lý, đánh giá, checkpoint, plots
v5: subfolder structure, bond detection plots, scatter_raw for all models
"""
import os, sys, signal, json, pickle, logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (CHECKPOINT_DIR, RESULTS_DIR, PREDICTIONS_DIR,
                    ANALYSIS_DIR, COMPARISON_DIR, LABEL_COLS, SPLIT_SEED,
                    get_model_dir)

LASER_NM = 784.815734863281

# ─── Logging ──────────────────────────────────────────────────────────────────
def get_logger(name):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                                         datefmt="%H:%M:%S"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg

log = get_logger("utils")

def _is_numeric(v):
    try: float(str(v)); return True
    except: return False

# ─── Wavelength → Raman shift ─────────────────────────────────────────────────
def wavelength_to_raman_shift(wl_nm, laser_nm=LASER_NM):
    return ((1.0/laser_nm) - (1.0/np.asarray(wl_nm, dtype=np.float64))) * 1e7

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(filepath, convert_wavelength=True):
    log.info(f"Đọc dữ liệu: {filepath}")
    df = pd.read_csv(filepath)
    log.info(f"  Shape raw: {df.shape}")

    KNOWN = {"sample id","sample","sample_id","sampleid","vial","vial#",
             "vial #","vial id","vialid","well","tube","name","group","id"}
    sample_col = None
    for c in df.columns:
        norm = c.lower().replace("_"," ").replace("-"," ").replace("#","").strip()
        if norm in KNOWN or c.lower().strip() in KNOWN:
            sample_col = c; break
    if sample_col is None:
        for c in df.columns:
            vals = df[c].dropna().astype(str).values[:20]
            if sum(1 for v in vals if not _is_numeric(v)) > len(vals)*0.7:
                sample_col = c; break
    if sample_col is None:
        raise ValueError("Không tìm thấy cột sample ID")
    log.info(f"  Sample col: '{sample_col}'")

    df_low = {c.lower().replace("_"," ").replace("-"," "): c for c in df.columns}
    label_cols = []
    for lc in LABEL_COLS:
        key = lc.lower().replace("_"," ").replace("-"," ")
        if key in df_low: label_cols.append(df_low[key])
        else:
            m = [o for l,o in df_low.items() if key in l or l in key]
            if m: label_cols.append(m[0])
    if len(label_cols) != 6:
        nc = df.select_dtypes(include=np.number).columns.tolist()
        label_cols = nc[-6:]; log.warning(f"  Fallback label cols: {label_cols}")
    log.info(f"  Label cols: {label_cols}")

    skip = set(label_cols) | {sample_col}
    spec_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    log.info(f"  Spectral cols: {len(spec_cols)}")

    X = df[spec_cols].values.astype(np.float32)
    Y = df[label_cols].values.astype(np.float32)
    sids = df[sample_col].values.astype(str)

    try: axis_vals = np.array([float(c) for c in spec_cols], dtype=np.float64)
    except: axis_vals = np.linspace(400, 1800, len(spec_cols), dtype=np.float64)

    ax_min, ax_max = axis_vals.min(), axis_vals.max()
    is_wl = (ax_min > 400 and ax_max < 1200 and ax_max - ax_min < 300)
    if convert_wavelength and is_wl:
        log.info(f"  Phát hiện trục Wavelength: {ax_min:.2f}–{ax_max:.2f} nm")
        log.info(f"  Chuyển đổi → Raman shift (laser={LASER_NM} nm)")
        wavenumbers = wavelength_to_raman_shift(axis_vals).astype(np.float32)
        if not np.all(np.diff(wavenumbers) >= 0):
            si = np.argsort(wavenumbers); wavenumbers = wavenumbers[si]; X = X[:, si]
    else:
        wavenumbers = axis_vals.astype(np.float32)
    log.info(f"  Raman shift: {wavenumbers.min():.1f}–{wavenumbers.max():.1f} cm⁻¹")

    rs = Y.sum(axis=1, keepdims=True); rs[rs==0] = 1
    Y = (Y / rs).astype(np.float32)
    log.info(f"  X={X.shape}, Y={Y.shape}")
    return X, Y, sids, wavenumbers


def aggregate_by_sample(X, Y, sids, method="median"):
    unique = list(dict.fromkeys(sids))
    Xl, Yl = [], []
    for sid in unique:
        mask = sids == sid
        Xl.append(np.median(X[mask], axis=0) if method=="median" else X[mask].mean(0))
        Yl.append(Y[mask][0])
    X_agg = np.array(Xl, dtype=np.float32)
    Y_agg = np.array(Yl, dtype=np.float32)
    log.info(f"  Aggregate ({method}): {len(sids)} phổ → {len(unique)} samples")
    return X_agg, Y_agg, np.array(unique)


def split_data(X, Y, sids, train_n=48, seed=SPLIT_SEED):
    unique = np.unique(sids); n = len(unique)
    log.info(f"  Split: {n} samples → train={train_n}, test={n-train_n}")
    rng = np.random.RandomState(seed); sh = rng.permutation(unique)
    tr_set = set(sh[:train_n]); te_set = set(sh[train_n:])
    tr_m = np.array([s in tr_set for s in sids])
    log.info(f"  Train: {tr_m.sum()} | Test: {(~tr_m).sum()}")
    return X[tr_m], X[~tr_m], Y[tr_m], Y[~tr_m], tr_set, te_set


# ─── Preprocessing ────────────────────────────────────────────────────────────
def remove_cosmic_rays(s, thr=5.0):
    s=s.copy(); d=np.diff(s); mu,sig=d.mean(),d.std()+1e-10
    for pos in (np.where(np.abs((d-mu)/sig)>thr)[0]+1):
        l,r=max(0,pos-1),min(len(s)-1,pos+1); s[pos]=(s[l]+s[r])/2.
    return s.astype(np.float32)

def savitzky_golay(s, window=9, poly=2):
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(s, window_length=window, polyorder=poly).astype(np.float32)
    except:
        k=window//2; out=s.copy()
        for i in range(k,len(s)-k): out[i]=s[i-k:i+k+1].mean()
        return out.astype(np.float32)

def snip_baseline(s, max_iter=20):
    w=np.sqrt(np.sqrt(np.maximum(s,0)+1))
    for p in range(1,max_iter+1):
        wn=w.copy()
        for k in range(p,len(w)-p): wn[k]=min(w[k],(w[k-p]+w[k+p])/2.)
        w=wn
    return np.maximum(s-w**4,0.).astype(np.float32)

def als_baseline(s, lam=1e5, p=0.01, n_iter=20):
    try:
        from scipy import sparse; from scipy.sparse.linalg import spsolve
        n=len(s); D=sparse.diags([1,-2,1],[0,1,2],shape=(n-2,n))
        H=lam*D.T.dot(D); w=np.ones(n); z=s.copy()
        for _ in range(n_iter):
            z=spsolve(sparse.diags(w,0)+H,w*s); w=np.where(s>z,p,1-p)
        return np.maximum(s-z,0.).astype(np.float32)
    except: return np.maximum(s-s.min(),0.).astype(np.float32)

def snv_normalization(s):
    return ((s-s.mean())/(s.std()+1e-10)).astype(np.float32)

def area_normalization(s):
    return (s/(np.abs(s).sum()+1e-10)).astype(np.float32)

def msc_correction(X, ref=None):
    if ref is None: ref=X.mean(axis=0)
    out=np.zeros_like(X)
    for i,s in enumerate(X):
        b,a=np.polyfit(ref,s,1); out[i]=((s-a)/(b+1e-10)).astype(np.float32)
    return out

def polynomial_baseline(s, degree=3):
    x=np.arange(len(s),dtype=float)
    try: return np.maximum(s-np.polyval(np.polyfit(x,s,degree),x),0.).astype(np.float32)
    except: return s

def preprocess_batch(X, method="full"):
    out=X.copy().astype(np.float32)
    if method=="snv":       return np.array([snv_normalization(s) for s in out])
    elif method in ("als","als_snv"):
        out=np.array([als_baseline(s) for s in out])
        return np.array([snv_normalization(s) for s in out])
    elif method=="snip":
        out=np.array([snip_baseline(s) for s in out])
        return np.array([snv_normalization(s) for s in out])
    elif method=="msc":
        out=msc_correction(out); return np.array([snv_normalization(s) for s in out])
    elif method=="area":    return np.array([area_normalization(s) for s in out])
    elif method=="sg_snv":
        out=np.array([savitzky_golay(s) for s in out])
        out=np.array([polynomial_baseline(s,3) for s in out])
        return np.array([snv_normalization(s) for s in out])
    elif method=="cosmic_snip_snv":
        out=np.array([remove_cosmic_rays(s) for s in out])
        out=np.array([savitzky_golay(s) for s in out])
        out=np.array([snip_baseline(s) for s in out])
        return np.array([snv_normalization(s) for s in out])
    elif method=="cosmic_sg_area":
        out=np.array([remove_cosmic_rays(s) for s in out])
        out=np.array([savitzky_golay(s) for s in out])
        return np.array([area_normalization(s) for s in out])
    elif method=="full":
        out=np.array([remove_cosmic_rays(s) for s in out])
        out=np.array([als_baseline(s) for s in out])
        out=np.array([savitzky_golay(s,window=9,poly=2) for s in out])
        return np.array([snv_normalization(s) for s in out])
    elif method=="none": return out
    else: return preprocess_batch(X,"full")


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(Y_true, Y_pred, label_names=None):
    if label_names is None: label_names=[f"out{i}" for i in range(Y_true.shape[1])]
    r2=float(r2_score(Y_true,Y_pred,multioutput="uniform_average"))
    mae=float(mean_absolute_error(Y_true,Y_pred))
    mse=float(mean_squared_error(Y_true,Y_pred))
    per={nm:{"r2":float(r2_score(Y_true[:,j],Y_pred[:,j])),
             "mae":float(mean_absolute_error(Y_true[:,j],Y_pred[:,j])),
             "rmse":float(np.sqrt(mean_squared_error(Y_true[:,j],Y_pred[:,j])))}
         for j,nm in enumerate(label_names)}
    return {"r2":r2,"mae":mae,"mse":mse,"rmse":float(np.sqrt(mse)),"per_output":per}

def print_metrics(m, name="", indent=0):
    p=" "*indent
    print(f"{p}{'─'*50}")
    if name: print(f"{p} {name}")
    print(f"{p}  R²   = {m['r2']:.4f}\n{p}  MAE  = {m['mae']:.4f}")
    print(f"{p}  RMSE = {m['rmse']:.4f}\n{p}  MSE  = {m['mse']:.6f}")
    if "per_output" in m:
        for k,v in m["per_output"].items():
            print(f"{p}    {k:<20}: R²={v['r2']:.4f}  MAE={v['mae']:.4f}")
    print(f"{p}{'─'*50}")


# ─── Checkpoint ───────────────────────────────────────────────────────────────
def ckpt_path(mid, suf="best"):
    return os.path.join(CHECKPOINT_DIR, f"model{mid:02d}_{suf}.pkl")

def save_checkpoint(mid, payload, suf="best"):
    with open(ckpt_path(mid,suf),"wb") as f: pickle.dump(payload,f)

def load_checkpoint(mid, suf="best"):
    p=ckpt_path(mid,suf)
    if not os.path.exists(p): return None
    try:
        with open(p,"rb") as f: return pickle.load(f)
    except: return None

def save_torch_checkpoint(mid, state, suf="best"):
    import torch; torch.save(state,ckpt_path(mid,suf))

def load_torch_checkpoint(mid, suf="best"):
    import torch; p=ckpt_path(mid,suf)
    if not os.path.exists(p): return None
    try: return torch.load(p,map_location="cpu",weights_only=False)
    except: return None

def save_results(mid, metrics, predictions=None, extra=None):
    res={"model_id":mid,"metrics":metrics}
    if extra: res.update(extra)
    # Save in both root results and model subfolder
    model_dir = get_model_dir(mid)
    path = os.path.join(model_dir, f"model{mid:02d}_results.json")
    with open(path,"w") as f:
        json.dump(res,f,indent=2,
                  default=lambda x:float(x) if isinstance(x,(np.floating,np.integer)) else str(x))
    if predictions:
        np.savez(os.path.join(model_dir,f"model{mid:02d}_predictions.npz"),**predictions)
    log.info(f"  Đã lưu kết quả: {path}")

def load_results(mid):
    # Try new subfolder location first, then old
    for path in [
        os.path.join(get_model_dir(mid), f"model{mid:02d}_results.json"),
        os.path.join(RESULTS_DIR, f"model{mid:02d}_results.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    return None


# ─── Graceful Interrupt ───────────────────────────────────────────────────────
class GracefulInterrupt:
    def __init__(self):
        self.stop_requested=False; self._count=0
        signal.signal(signal.SIGINT,self._handler)
        signal.signal(signal.SIGTERM,self._handler)
    def _handler(self,signum,frame):
        self._count+=1
        if self._count==1: print("\n[!] Ctrl+C — dừng sau epoch. Ctrl+C lần 2 ngay."); self.stop_requested=True
        else: print("\n[!] Dừng ngay!"); sys.exit(1)
    def reset(self): self.stop_requested=False; self._count=0

INTERRUPTER=GracefulInterrupt()


# ─── Plot: Loss Curves ────────────────────────────────────────────────────────
def plot_loss_curves(history, mid, name):
    model_dir = get_model_dir(mid)
    keys=list(history.keys())
    fig,axes=plt.subplots(1,len(keys),figsize=(6*len(keys),4))
    if len(keys)==1: axes=[axes]
    for ax,key in zip(axes,keys):
        v=history[key]
        tr=v.get("train",[])
        ax.plot(tr,label="Train",lw=1.5)
        if "val" in v: ax.plot(v["val"],label="Val",lw=1.5)
        ax.set_title(f"M{mid} — {key}"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir,f"model{mid:02d}_loss.png"),dpi=100,bbox_inches="tight")
    plt.close()


def plot_r2_curve(r2_vals, mid, name):
    """Thay thế loss curve cho các model không có epochs (R² curve)"""
    model_dir = get_model_dir(mid)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(r2_vals,marker="o",lw=2,markersize=4,color="steelblue")
    ax.axhline(0,color="gray",ls="--",alpha=0.5)
    ax.set_xlabel("Step / Iteration"); ax.set_ylabel("R²")
    ax.set_title(f"Model {mid}: {name} — R² Curve"); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir,f"model{mid:02d}_loss.png"),dpi=100,bbox_inches="tight")
    plt.close()


# ─── Plot: Scatter (aggregated 6 samples) ─────────────────────────────────────
def plot_predictions(Y_true, Y_pred, mid, name, label_names=None):
    model_dir = get_model_dir(mid)
    if label_names is None: label_names=LABEL_COLS
    n=len(label_names); ncols=3; nrows=(n+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4*nrows)); axes=axes.flatten()
    for j,(ln,ax) in enumerate(zip(label_names,axes)):
        r2j=r2_score(Y_true[:,j],Y_pred[:,j])
        ax.scatter(Y_true[:,j],Y_pred[:,j],alpha=0.8,s=60)
        mn=min(Y_true[:,j].min(),Y_pred[:,j].min()); mx=max(Y_true[:,j].max(),Y_pred[:,j].max())
        ax.plot([mn,mx],[mn,mx],"r--",lw=1.5)
        ax.set_xlabel("True"); ax.set_ylabel("Pred")
        ax.set_title(f"{ln}\n$R^2$={r2j:.3f}"); ax.grid(True,alpha=0.3)
    for ax in axes[n:]: ax.set_visible(False)
    plt.suptitle(f"Model {mid}: {name}",fontsize=12,y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir,f"model{mid:02d}_scatter.png"),dpi=100,bbox_inches="tight")
    plt.close()


# ─── Plot: Scatter raw spectra (6 cụm × 90 điểm) ─────────────────────────────
def plot_predictions_raw(Y_true_raw, Y_pred_raw, sid_raw, mid, name, label_names=None):
    """Scatter với TẤT CẢ raw test spectra — 6 cụm điểm màu khác nhau"""
    model_dir = get_model_dir(mid)
    if label_names is None: label_names=LABEL_COLS
    n=len(label_names); ncols=3; nrows=(n+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(5*ncols,4*nrows)); axes=axes.flatten()
    unique_sids=list(dict.fromkeys(sid_raw))
    cmap=plt.cm.tab10(np.linspace(0,0.9,len(unique_sids)))
    color_map={s:c for s,c in zip(unique_sids,cmap)}

    for j,(ln,ax) in enumerate(zip(label_names,axes)):
        r2j=r2_score(Y_true_raw[:,j],Y_pred_raw[:,j])
        for sid in unique_sids:
            mask=sid_raw==sid
            ax.scatter(Y_true_raw[mask,j],Y_pred_raw[mask,j],
                       alpha=0.35,s=15,c=[color_map[sid]],label=sid)
        mn=min(Y_true_raw[:,j].min(),Y_pred_raw[:,j].min())
        mx=max(Y_true_raw[:,j].max(),Y_pred_raw[:,j].max())
        ax.plot([mn,mx],[mn,mx],"r--",lw=1.5)
        ax.set_xlabel("True"); ax.set_ylabel("Pred")
        ax.set_title(f"{ln}\n$R^2$={r2j:.3f}"); ax.grid(True,alpha=0.3)
    for ax in axes[n:]: ax.set_visible(False)
    # Legend
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower right",ncol=3,fontsize=7,title="Sample",
               bbox_to_anchor=(1.0,0.0))
    plt.suptitle(f"Model {mid}: {name}  (N={len(Y_true_raw)} raw spectra, 6 clusters)",
                 fontsize=11,y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir,f"model{mid:02d}_scatter_raw.png"),
                dpi=100,bbox_inches="tight")
    plt.close()


# ─── Plot: Bond Detection on Spectrum ─────────────────────────────────────────
def plot_bond_detect(spectrum_proc: np.ndarray,
                     wavenumbers: np.ndarray,
                     sample_id: str,
                     model_id: int,
                     model_name: str,
                     composition: Optional[np.ndarray] = None,
                     label_names: Optional[List[str]] = None):
    """
    Vẽ phổ đã tiền xử lý + phát hiện đỉnh + bounding box + nhãn liên kết.
    Lưu vào model subfolder.
    """
    from scipy.signal import find_peaks
    from spectral_knowledge import get_active_regions, get_region_indices

    model_dir = get_model_dir(model_id)
    regions   = get_active_regions(wavenumbers)
    ridx      = get_region_indices(wavenumbers, regions)
    bond_colors = {
        "amine":       "#E8593C",
        "amine_bend":  "#E8593C",
        "carboxylate": "#3B8BD4",
        "amide_I":     "#534AB7",
        "amide_III":   "#9F77DD",
        "aliphatic":   "#888780",
        "aromatic":    "#EF9F27",
        "amino_frame": "#1D9E75",
        "polar":       "#D4537E",
        "skeletal":    "#5DCAA5",
        "C-hetero":    "#C0DD97",
        "ring":        "#FAC775",
        "unknown":     "#B4B2A9",
    }

    # ── Tìm đỉnh ─────────────────────────────────────────────────────────────
    spec_pos = np.maximum(spectrum_proc, 0.)
    min_height = spec_pos.max() * 0.05
    min_prom   = spec_pos.max() * 0.03
    peaks, props = find_peaks(spec_pos,
                              height=min_height,
                              prominence=min_prom,
                              distance=15,
                              width=3)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wavenumbers, spec_pos, lw=1.2, color="#444", zorder=2, label="Spectrum")
    ax.fill_between(wavenumbers, spec_pos, alpha=0.08, color="#444")

    # ── Vẽ bounding box cho mỗi bond region ───────────────────────────────────
    y_max = spec_pos.max()
    labeled_regions = set()
    for br in regions:
        i0, i1 = ridx[br.name]
        if i1 <= i0: continue
        seg = spec_pos[i0:i1]
        if seg.max() < min_height * 0.3: continue
        color = bond_colors.get(br.bond_type, "#B4B2A9")
        wn0, wn1 = wavenumbers[i0], wavenumbers[i1-1]
        seg_h = seg.max()
        rect = plt.Rectangle((wn0, -0.02*y_max),
                              wn1-wn0, seg_h + 0.07*y_max,
                              linewidth=1.5, edgecolor=color,
                              facecolor=color, alpha=0.08, zorder=1)
        ax.add_patch(rect)
        # Label box at top
        ax.text((wn0+wn1)/2, seg_h + 0.09*y_max,
                f"{br.name}\n({int((wn0+wn1)/2)} cm⁻¹)",
                ha="center", va="bottom", fontsize=7,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8, lw=0.8))
        labeled_regions.add(br.name)

    # ── Marcar đỉnh với vị trí chính xác ─────────────────────────────────────
    for pk in peaks:
        wn_pk = wavenumbers[pk]
        h_pk  = spec_pos[pk]
        # Tìm bond region chứa đỉnh này
        bond_name = "?"
        bond_color = "#333"
        for br in regions:
            i0, i1 = ridx[br.name]
            if i0 <= pk < i1:
                bond_name  = br.name
                bond_color = bond_colors.get(br.bond_type, "#333")
                break
        ax.plot(wn_pk, h_pk, "v", color=bond_color, markersize=7, zorder=5)
        ax.annotate(f"{wn_pk:.0f}",
                    xy=(wn_pk, h_pk), xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", fontsize=7.5, color=bond_color,
                    arrowprops=dict(arrowstyle="-", color=bond_color, lw=0.8))

    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity (preprocessed)", fontsize=11)
    ax.set_title(f"Model {model_id} ({model_name}) — Sample: {sample_id}\n"
                 f"Preprocessed spectrum + Bond detection ({len(peaks)} peaks found)",
                 fontsize=11)
    ax.set_xlim(wavenumbers[0]-20, wavenumbers[-1]+20)
    ax.set_ylim(-0.05*y_max, y_max * 1.25)
    ax.grid(True, alpha=0.2)

    # ── Thêm composition nếu biết ─────────────────────────────────────────────
    if composition is not None and label_names is not None:
        comp_str = "  ".join(f"{n[:3]}:{v:.2f}"
                             for n,v in zip(label_names,composition))
        ax.text(0.01, 0.97, f"Composition: {comp_str}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color="#555",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ── Legend bond regions ────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    seen = {}
    for br in regions:
        bt = br.bond_type
        if bt not in seen: seen[bt] = bond_colors.get(bt,"#B4B2A9")
    legend_els = [Patch(color=c, alpha=0.6, label=bt) for bt,c in list(seen.items())[:8]]
    ax.legend(handles=legend_els, fontsize=7, loc="upper right",
              ncol=2, title="Bond types", title_fontsize=7)

    plt.tight_layout()
    safe_sid = sample_id.replace("/","_").replace("\\","_")
    save_path = os.path.join(model_dir, f"model{model_id:02d}_bonddetect_{safe_sid}.png")
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    return save_path


def plot_bond_detect_batch(X_proc: np.ndarray,
                            wavenumbers: np.ndarray,
                            sample_ids,
                            model_id: int,
                            model_name: str,
                            Y: Optional[np.ndarray] = None,
                            label_names: Optional[List[str]] = None):
    """Vẽ bond detection cho batch spectra (6 test samples)"""
    saved = []
    for i, (s, sid) in enumerate(zip(X_proc, sample_ids)):
        comp = Y[i] if Y is not None else None
        p = plot_bond_detect(s, wavenumbers, str(sid),
                             model_id, model_name, comp, label_names)
        saved.append(p)
    log.info(f"  ✓ Đã lưu {len(saved)} bond detection plots")
    return saved


# ─── Plot: Comparison (comparison folder) ────────────────────────────────────
def plot_comparison(all_results):
    ids=[]; r2s=[]; maes=[]; names=[]
    for i in sorted(all_results.keys()):
        r=all_results[i]; m=r.get("metrics",{})
        r2=m.get("r2",float("nan")); mae=m.get("mae",float("nan"))
        if not np.isnan(r2):
            ids.append(i); r2s.append(r2); maes.append(mae)
            names.append(r.get("name",f"M{i}"))
    if not ids: return
    colors=plt.cm.viridis(np.linspace(0.2,0.9,len(ids)))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
    b1=ax1.barh(names,r2s,color=colors)
    ax1.set_xlim(-2.0,1.05); ax1.axvline(0,color="gray",ls="--",alpha=0.5)
    ax1.set_xlabel("R²"); ax1.set_title("R² Score (↑ tốt hơn)")
    for bar,v in zip(b1,r2s): ax1.text(max(v+0.02,-1.9),bar.get_y()+bar.get_height()/2,f"{v:.3f}",va="center",fontsize=8)
    b2=ax2.barh(names,maes,color=colors)
    ax2.set_xlabel("MAE"); ax2.set_title("MAE (↓ tốt hơn)")
    for bar,v in zip(b2,maes): ax2.text(v+0.001,bar.get_y()+bar.get_height()/2,f"{v:.4f}",va="center",fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR,"comparison.png"),dpi=120,bbox_inches="tight")
    plt.close()


# ─── PyTorch helpers ──────────────────────────────────────────────────────────
def get_device():
    import torch; return torch.device("cpu")

def make_dataloader(X,Y,bs=32,shuffle=True):
    import torch; from torch.utils.data import TensorDataset,DataLoader
    return DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32),
                                    torch.tensor(Y,dtype=torch.float32)),
                      batch_size=bs,shuffle=shuffle,num_workers=0)

def kl_div_loss(pred,target):
    import torch.nn.functional as F
    return F.kl_div(pred.clamp(min=1e-8).log(),target.clamp(min=1e-8),reduction="batchmean")

def dirichlet_normalize(Y):
    Y=np.maximum(Y,0.); s=Y.sum(axis=1,keepdims=True); s[s==0]=1
    return (Y/s).astype(np.float32)

def information_score(features):
    from sklearn.decomposition import PCA
    n,d=features.shape
    if d>n: features=features[:,:n]
    try:
        evr=PCA().fit(features).explained_variance_ratio_
        evr=evr[evr>1e-10]
        return float(np.exp(-np.sum(evr*np.log(evr+1e-10))))
    except: return 0.

def mutual_info_score(features,Y):
    sc=[]
    for j in range(Y.shape[1]):
        for d in range(min(features.shape[1],10)):
            c=np.corrcoef(features[:,d],Y[:,j])[0,1]
            if not np.isnan(c): sc.append(abs(c))
    return float(np.mean(sc)) if sc else 0.


def plot_summary_table(all_results, save_dir=None):
    """Vẽ bảng tổng hợp kết quả tất cả model thành ảnh"""
    if save_dir is None: save_dir = COMPARISON_DIR
    from config import MODEL_NAMES as MN
    rows = []
    for mid in sorted(all_results.keys()):
        r = all_results[mid]; m = r.get("metrics", {})
        rows.append({
            "id": mid,
            "name": r.get("name", MN.get(mid, f"M{mid}"))[:28],
            "r2":   m.get("r2", float("nan")),
            "mae":  m.get("mae", float("nan")),
            "rmse": m.get("rmse", float("nan")),
            "time": r.get("elapsed_s", float("nan")),
        })
    if not rows: return
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, max(3.5, n*0.55+1.5)))
    ax.axis("off")
    col_labels = ["#", "Model", "R²", "MAE", "RMSE", "Time (s)"]
    col_widths = [0.04, 0.36, 0.14, 0.12, 0.14, 0.12]

    # Header
    header_y = 0.97; row_h = 0.88 / (n+1)
    colors_h = ["#2C2C2A"] * len(col_labels)
    x_cur = 0.02
    for label, w, col in zip(col_labels, col_widths, colors_h):
        ax.text(x_cur + w/2, header_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.3", fc="#2C2C2A", ec="none"))
        x_cur += w

    # Rows
    best_r2 = max((r["r2"] for r in rows if not np.isnan(r["r2"])), default=0.)
    for i, row in enumerate(rows):
        y = header_y - (i+1)*row_h
        bg = "#EAF3DE" if row["r2"] == best_r2 else ("#F1EFE8" if i%2==0 else "white")
        x_cur = 0.02
        vals = [str(row["id"]), row["name"],
                f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "—",
                f"{row['mae']:.4f}" if not np.isnan(row['mae']) else "—",
                f"{row['rmse']:.4f}" if not np.isnan(row['rmse']) else "—",
                f"{row['time']:.1f}s" if not np.isnan(row['time']) else "—"]
        aligns = ["center","left","center","center","center","center"]
        for val, w, ha in zip(vals, col_widths, aligns):
            fc = "#1D9E75" if (val==str(row['id']) and row['r2']==best_r2) else bg
            ax.text(x_cur + (0.02 if ha=="left" else w/2), y, val,
                    ha=ha, va="center", fontsize=9, transform=ax.transAxes,
                    bbox=dict(boxstyle="square,pad=0.25", fc=fc, ec="#E0E0E0", lw=0.5))
            x_cur += w

    ax.text(0.5, 0.01, f"★ Best model highlighted in green  |  {len(rows)} models evaluated",
            ha="center", va="bottom", fontsize=8, color="#888", transform=ax.transAxes)
    plt.title("Kết quả Đánh giá Tất cả Mô hình — Raman AA Analysis",
              fontsize=12, fontweight="bold", pad=10)
    out = os.path.join(save_dir, "models_summary_table.png")
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    log.info(f"  Đã lưu bảng tổng hợp: {out}")
