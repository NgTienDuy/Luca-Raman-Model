"""
Utility functions: data loading, preprocessing, evaluation, plotting.
"""
import os, gc, json, logging, time, warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path=None):
    """Load CSV, auto-detect wavelength columns, convert to wavenumbers."""
    path = path or C.DATA_PATH
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # Identify spectral columns (numeric headers that are wavelengths)
    spec_cols = [c for c in df.columns if c not in
                 (["vial #"] + C.AA_NAMES)]
    wavelengths_nm = np.array([float(c) for c in spec_cols])

    # Convert to Raman shift
    wavenumbers = (1.0 / C.LASER_WL_NM - 1.0 / wavelengths_nm) * 1e7

    X = df[spec_cols].values.astype(np.float32)
    Y = df[C.AA_NAMES].values.astype(np.float32)
    sample_ids = df["vial #"].values

    log.info(f"  X shape: {X.shape}, wavenumber range: {wavenumbers.min():.1f}–{wavenumbers.max():.1f} cm⁻¹")
    log.info(f"  Unique samples: {len(np.unique(sample_ids))}")
    return X, Y, wavenumbers, sample_ids


def split_data(X, Y, sample_ids, seed=None):
    """Group-stratified split by sample ID."""
    seed = seed or C.SEED
    np.random.seed(seed)

    unique_sids = np.unique(sample_ids)
    n = len(unique_sids)

    # Two-stage split: first test, then val from remaining
    gss1 = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE / n, random_state=seed)
    rest_idx, test_idx = next(gss1.split(X, Y, groups=sample_ids))

    X_rest, Y_rest, sid_rest = X[rest_idx], Y[rest_idx], sample_ids[rest_idx]
    rest_unique = np.unique(sid_rest)
    n_rest = len(rest_unique)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=C.VAL_SIZE / n_rest, random_state=seed)
    train_idx2, val_idx2 = next(gss2.split(X_rest, Y_rest, groups=sid_rest))

    train_idx_global = rest_idx[train_idx2]
    val_idx_global = rest_idx[val_idx2]
    test_idx_global = test_idx

    def _get(idx):
        return X[idx], Y[idx], sample_ids[idx]

    Xtr, Ytr, sid_tr = _get(train_idx_global)
    Xv, Yv, sid_v = _get(val_idx_global)
    Xt, Yt, sid_t = _get(test_idx_global)

    log.info(f"  Split: train={len(Xtr)} ({len(np.unique(sid_tr))} samples), "
             f"val={len(Xv)} ({len(np.unique(sid_v))} samples), "
             f"test={len(Xt)} ({len(np.unique(sid_t))} samples)")
    return Xtr, Xv, Xt, Ytr, Yv, Yt, sid_tr, sid_v, sid_t


# ═══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def remove_cosmic_rays(s, thr=None):
    """Remove cosmic ray spikes via 1st-difference detection."""
    thr = thr or C.COSMIC_THR
    s = s.copy()
    d = np.diff(s)
    med, std = np.median(np.abs(d)), np.std(d)
    spikes = np.where(np.abs(d) > med + thr * std)[0]
    for i in spikes:
        lo = max(0, i - 2)
        hi = min(len(s) - 1, i + 2)
        s[i] = np.mean([s[lo], s[hi]])
    return s


def snip_baseline(s, max_iter=None):
    """SNIP iterative peak-stripping baseline."""
    max_iter = max_iter or C.SNIP_ITER
    y = np.log(np.log(np.sqrt(np.maximum(s, 1e-10) + 1) + 1) + 1)
    for i in range(1, max_iter + 1):
        rolled_lo = np.roll(y, i)
        rolled_hi = np.roll(y, -i)
        avg = (rolled_lo + rolled_hi) / 2.0
        y = np.minimum(y, avg)
    # Inverse transform
    baseline = (np.exp(np.exp(y) - 1) - 1) ** 2 - 1
    return np.maximum(s - baseline, 0)


def als_baseline(s, lam=None, p=None, niter=10):
    """Asymmetric Least Squares baseline correction."""
    lam = lam or C.ALS_LAM
    p = p or C.ALS_P
    L = len(s)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2)).toarray()
    H = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0)
        Z = W + H
        z = spsolve(Z.tocsc() if hasattr(Z, 'tocsc') else Z, w * s)
        w = p * (s > z) + (1 - p) * (s <= z)
    return np.maximum(s - z, 0)


def savitzky_golay(s, window=None, poly=None):
    """Savitzky-Golay smoothing."""
    window = window or C.SG_WINDOW
    poly = poly or C.SG_POLY
    return savgol_filter(s, window, poly)


def snv_normalization(s):
    """Standard Normal Variate."""
    m, sd = s.mean(), s.std()
    if sd < 1e-10:
        return s - m
    return (s - m) / sd


def area_normalization(s):
    """Divide by total area."""
    area = np.abs(s).sum()
    if area < 1e-10:
        return s
    return s / area


def msc_correction(X, ref=None):
    """Multiplicative Scatter Correction."""
    if ref is None:
        ref = X.mean(axis=0)
    X_corr = np.zeros_like(X)
    for i in range(len(X)):
        coef = np.polyfit(ref, X[i], 1)
        X_corr[i] = (X[i] - coef[1]) / max(coef[0], 1e-10)
    return X_corr


def preprocess_single(s, method='full'):
    """Preprocess a single spectrum."""
    if method == 'none':
        return s.copy()
    elif method == 'snv':
        return snv_normalization(s)
    elif method == 'full':
        s = remove_cosmic_rays(s)
        s = snip_baseline(s)
        s = savitzky_golay(s)
        s = snv_normalization(s)
        return s
    elif method == 'sg_snv':
        s = savitzky_golay(s)
        s = snv_normalization(s)
        return s
    elif method == 'als_snv':
        s = als_baseline(s)
        s = snv_normalization(s)
        return s
    else:
        return s.copy()


def preprocess_batch(X, method='full'):
    """Apply named preprocessing pipeline to batch."""
    return np.array([preprocess_single(x, method) for x in X], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(Y_true, Y_pred, n_features=None):
    """Compute all metrics."""
    n = len(Y_true)
    # For adjusted R², use effective df.  Cap at n/3 to avoid degeneracy
    p = min(n_features or 50, n // 3)

    # Per amino acid
    per_aa = {}
    for i, name in enumerate(C.AA_NAMES):
        yt, yp = Y_true[:, i], Y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_i = 1 - ss_res / max(ss_tot, 1e-10)
        mae_i = np.mean(np.abs(yt - yp))
        per_aa[name] = {"R2": float(r2_i), "MAE": float(mae_i)}

    # Overall
    r2_list = [per_aa[n]["R2"] for n in C.AA_NAMES]
    r2 = float(np.mean(r2_list))
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    residuals = Y_true - Y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE (avoid division by zero)
    mask = Y_true > 1e-6
    if mask.sum() > 0:
        mape = float(np.mean(np.abs(residuals[mask] / Y_true[mask])) * 100)
    else:
        mape = 0.0

    # RMSLE
    rmsle = float(np.sqrt(np.mean((np.log1p(np.maximum(Y_pred, 0)) -
                                    np.log1p(np.maximum(Y_true, 0))) ** 2)))
    max_err = float(np.max(np.abs(residuals)))

    return {
        "R2": r2, "Adj_R2": adj_r2, "MAE": mae, "MSE": mse,
        "RMSE": rmse, "MAPE": mape, "RMSLE": rmsle, "MaxError": max_err,
        "per_aa": per_aa
    }


def median_predictions(Y_pred, sample_ids):
    """Get median prediction per sample."""
    unique = np.unique(sample_ids)
    Y_med = np.zeros((len(unique), Y_pred.shape[1]))
    for i, sid in enumerate(unique):
        mask = sample_ids == sid
        Y_med[i] = np.median(Y_pred[mask], axis=0)
    return Y_med, unique


def aggregate_by_sample(X, Y, sample_ids):
    """Compute median spectrum and label per sample."""
    unique = np.unique(sample_ids)
    Xm = np.zeros((len(unique), X.shape[1]), dtype=np.float32)
    Ym = np.zeros((len(unique), Y.shape[1]), dtype=np.float32)
    for i, s in enumerate(unique):
        mask = sample_ids == s
        Xm[i] = np.median(X[mask], axis=0)
        Ym[i] = np.median(Y[mask], axis=0)
    return Xm, Ym, unique


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def softmax_normalize(Y):
    """Apply softmax-style normalization (clip + renorm)."""
    Y = np.maximum(Y, 0)
    row_sums = Y.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return Y / row_sums


def plot_scatter_aggregated(Y_true, Y_pred, sample_ids, model_num, save_dir):
    """Scatter plot: 6 aggregated data points (median per sample)."""
    unique_sids = np.unique(sample_ids)
    n_samples = len(unique_sids)
    Y_pred_med = np.zeros((n_samples, C.NUM_AA))
    Y_true_med = np.zeros((n_samples, C.NUM_AA))

    for i, sid in enumerate(unique_sids):
        mask = sample_ids == sid
        Y_pred_med[i] = np.median(Y_pred[mask], axis=0)
        Y_true_med[i] = np.median(Y_true[mask], axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for i, name in enumerate(C.AA_NAMES):
        ax = axes[i]
        yt, yp = Y_true_med[:, i], Y_pred_med[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        ax.scatter(yt, yp, c='steelblue', s=80, zorder=3, edgecolors='k', linewidth=0.5)
        lo = min(yt.min(), yp.min()) - 0.05
        hi = max(yt.max(), yp.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  R²={r2:.3f}")
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f"Model {model_num:02d} — Aggregated (N={n_samples} samples)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, f"model{model_num:02d}_scatter.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")


def plot_scatter_raw(Y_true, Y_pred, sample_ids, model_num, save_dir):
    """Scatter plot: all individual spectra, coloured by sample ID."""
    unique_sids = np.unique(sample_ids)
    n_samples = len(unique_sids)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_samples, 10)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for i, name in enumerate(C.AA_NAMES):
        ax = axes[i]
        for j, sid in enumerate(unique_sids):
            mask = sample_ids == sid
            yt = Y_true[mask, i]
            yp = Y_pred[mask, i]
            ax.scatter(yt, yp, c=[colors[j % len(colors)]], s=12, alpha=0.6,
                       label=str(sid) if i == 0 else None)

        all_vals = np.concatenate([Y_true[:, i], Y_pred[:, i]])
        lo, hi = all_vals.min() - 0.05, all_vals.max() + 0.05
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}")

    # Legend on first subplot
    if n_samples <= 10:
        axes[0].legend(fontsize=6, loc='upper left')

    fig.suptitle(f"Model {model_num:02d} — Raw (N={len(Y_true)}, {n_samples} clusters)", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, f"model{model_num:02d}_scatter_raw.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")


def plot_loss_curve(train_losses, val_losses, model_num, save_dir):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train', linewidth=1.5)
    ax.plot(val_losses, label='Validation', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Model {model_num:02d} — Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, f"model{model_num:02d}_loss.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")


def plot_sweep(param_vals, scores, param_name, model_num, save_dir):
    """Plot hyperparameter sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_vals, scores, 'o-', linewidth=1.5, markersize=6)
    ax.set_xlabel(param_name)
    ax.set_ylabel('R²')
    ax.set_title(f'Model {model_num:02d} — Hyperparameter Sweep')
    ax.grid(True, alpha=0.3)
    if all(v > 0 for v in param_vals):
        ax.set_xscale('log')
    plt.tight_layout()
    out = os.path.join(save_dir, f"model{model_num:02d}_sweep.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")


def save_results(metrics, model_num, save_dir, extra=None):
    """Save metrics to JSON."""
    data = {"model": model_num, "name": C.MODEL_NAMES.get(model_num, ""), **metrics}
    if extra:
        data.update(extra)
    out = os.path.join(save_dir, f"model{model_num:02d}_results.json")
    with open(out, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"  Saved {out}")


def set_seeds(seed=None):
    """Set all random seeds."""
    seed = seed or C.SEED
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
