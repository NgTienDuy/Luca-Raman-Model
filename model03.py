"""
Model 03: Two-Stage Hybrid — Linear unmixing + nonlinear correction.
Stage 1: NNLS using pure reference spectra → initial estimate.
Stage 2: MLP/GBR corrects residual.
"""
import os, time, logging, gc
import numpy as np
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import config as C
from utils import (preprocess_batch, compute_metrics, softmax_normalize, plot_scatter_aggregated, plot_scatter_raw,
                   plot_sweep, save_results, set_seeds)

log = logging.getLogger(__name__)
MODEL_NUM = 3

PURE_NAMES = {
    'DL-alanine': 0, 'L-asparagine': 1, 'L-aspartic-acid': 2,
    'L-glutamic-acid': 3, 'L-histidine': 4, 'D-glucosamine': 5
}


def _get_pure_refs(X, Y, sids):
    """Extract median pure-component reference spectra."""
    refs = np.zeros((6, X.shape[1]), dtype=np.float32)
    for name, idx in PURE_NAMES.items():
        mask = sids == name
        if mask.sum() > 0:
            refs[idx] = np.median(X[mask], axis=0)
        else:
            # If pure not in this split, use rows with Y[idx]≈1
            for i in range(len(Y)):
                if Y[i, idx] > 0.95:
                    mask2 = sids == sids[i]
                    refs[idx] = np.median(X[mask2], axis=0)
                    break
    return refs


def _nnls_unmix(spectrum, refs):
    """NNLS unmixing of a single spectrum."""
    # refs: (6, n_wl), spectrum: (n_wl,)
    coefs, _ = nnls(refs.T, spectrum)
    total = coefs.sum()
    if total > 1e-10:
        coefs = coefs / total
    else:
        coefs = np.ones(6) / 6
    return coefs


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)

    sid_train = kw.get("sid_train")
    sid_val = kw.get("sid_val")
    sid_test = kw.get("sid_test")

    log.info("M03: Preprocessing...")
    Xtr = preprocess_batch(X_train, 'full')
    Xv = preprocess_batch(X_val, 'full')
    Xt = preprocess_batch(X_test, 'full')

    # Get pure refs from training data
    log.info("M03: Extracting pure reference spectra...")
    refs = _get_pure_refs(Xtr, Y_train, sid_train)

    # Stage 1: NNLS unmixing for all sets
    log.info("M03: Stage 1 — NNLS unmixing...")
    Y1_train = np.array([_nnls_unmix(x, refs) for x in Xtr])
    Y1_val = np.array([_nnls_unmix(x, refs) for x in Xv])
    Y1_test = np.array([_nnls_unmix(x, refs) for x in Xt])

    # Residuals for training Stage 2
    residuals_train = Y_train - Y1_train

    # Stage 2: GBR on [spectrum_features + Y1] → residual
    log.info("M03: Stage 2 — Gradient Boosting correction...")
    # Use PCA-reduced features to keep it manageable
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30, random_state=C.SEED)
    Xtr_pca = pca.fit_transform(Xtr)
    Xv_pca = pca.transform(Xv)
    Xt_pca = pca.transform(Xt)

    # Combine: PCA features + Stage 1 predictions
    F_train = np.hstack([Xtr_pca, Y1_train])
    F_val = np.hstack([Xv_pca, Y1_val])
    F_test = np.hstack([Xt_pca, Y1_test])

    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)
    F_val_s = scaler.transform(F_val)
    F_test_s = scaler.transform(F_test)

    # Sweep n_estimators
    n_est_vals = [50, 100]
    sweep_scores = []
    best_r2 = -999
    best_model = None
    best_n = None

    for n_est in n_est_vals:
        gbr = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=n_est, max_depth=4,
                                     learning_rate=0.05, random_state=C.SEED)
        )
        gbr.fit(F_train_s, residuals_train)
        corr_val = gbr.predict(F_val_s)
        Y_pred_val = softmax_normalize(Y1_val + corr_val)
        met = compute_metrics(Y_val, Y_pred_val, F_train.shape[1])
        sweep_scores.append(met["R2"])
        if met["R2"] > best_r2:
            best_r2 = met["R2"]
            best_model = gbr
            best_n = n_est

    log.info(f"M03: Best n_estimators={best_n}, val R²={best_r2:.4f}")
    plot_sweep(n_est_vals, sweep_scores, "n_estimators (GBR)", MODEL_NUM, save_dir)

    # Predict on test
    corr_test = best_model.predict(F_test_s)
    Y_pred = softmax_normalize(Y1_test + corr_test)

    metrics = compute_metrics(Y_test, Y_pred, F_train.shape[1])
    metrics["best_n_estimators"] = best_n
    metrics["epochs"] = 0
    metrics["training_time"] = time.time() - t0

    log.info(f"M03: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    gc.collect()
    return Y_pred, metrics
