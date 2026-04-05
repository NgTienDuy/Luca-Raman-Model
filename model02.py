"""
Model 02: MCR-ALS (NMF) + PLSR — Classical chemometric unmixing.
"""
import os, time, logging, gc
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

import config as C
from utils import (preprocess_batch, compute_metrics, softmax_normalize, plot_scatter_aggregated, plot_scatter_raw,
                   plot_sweep, save_results, set_seeds)
from spectral_knowledge import extract_all_bond_features

log = logging.getLogger(__name__)
MODEL_NUM = 2


def _median_by_group(X, Y, sids):
    unique = np.unique(sids)
    Xm = np.zeros((len(unique), X.shape[1]), dtype=np.float32)
    Ym = np.zeros((len(unique), Y.shape[1]), dtype=np.float32)
    for i, s in enumerate(unique):
        mask = sids == s
        Xm[i] = np.median(X[mask], axis=0)
        Ym[i] = np.median(Y[mask], axis=0)
    return Xm, Ym, unique


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)

    sid_train = kw.get("sid_train")
    sid_val = kw.get("sid_val")
    sid_test = kw.get("sid_test")

    log.info("M02: Preprocessing...")
    Xtr = preprocess_batch(X_train, 'full')
    Xv = preprocess_batch(X_val, 'full')
    Xt = preprocess_batch(X_test, 'full')

    # Make non-negative for NMF
    Xtr_nn = np.maximum(Xtr, 0)
    Xv_nn = np.maximum(Xv, 0)
    Xt_nn = np.maximum(Xt, 0)

    # NMF
    log.info("M02: Fitting NMF...")
    nmf = NMF(n_components=C.NMF_K, max_iter=300, random_state=C.SEED, init='nndsvda')
    H_train = nmf.fit_transform(Xtr_nn)
    H_val = nmf.transform(Xv_nn)
    H_test = nmf.transform(Xt_nn)

    # Bond integrals
    log.info("M02: Extracting bond features...")
    bf_train = extract_all_bond_features(Xtr, wavenumbers)
    bf_val = extract_all_bond_features(Xv, wavenumbers)
    bf_test = extract_all_bond_features(Xt, wavenumbers)

    # Combine: [NMF concentrations | bond integrals]
    F_train = np.hstack([H_train, bf_train])
    F_val = np.hstack([H_val, bf_val])
    F_test = np.hstack([H_test, bf_test])

    scaler = StandardScaler()
    F_train_s = scaler.fit_transform(F_train)
    F_val_s = scaler.transform(F_val)
    F_test_s = scaler.transform(F_test)

    # Median for training PLSR
    Ftr_m, Ytr_m, _ = _median_by_group(F_train_s, Y_train, sid_train)
    Fv_m, Yv_m, _ = _median_by_group(F_val_s, Y_val, sid_val)

    # Sweep LV
    log.info("M02: Sweeping PLSR components...")
    max_lv = min(max(C.PLSR_LV_RANGE), Ftr_m.shape[0] - 1, Ftr_m.shape[1])
    lv_range = [lv for lv in C.PLSR_LV_RANGE if lv <= max_lv]

    sweep_scores = []
    best_r2 = -999
    best_pls = None
    best_lv = None

    for n_lv in lv_range:
        try:
            pls = PLSRegression(n_components=n_lv, max_iter=1000)
            pls.fit(Ftr_m, Ytr_m)
            pred = softmax_normalize(pls.predict(Fv_m))
            metrics = compute_metrics(Yv_m, pred, Ftr_m.shape[1])
            sweep_scores.append(metrics["R2"])
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_pls = pls
                best_lv = n_lv
        except Exception:
            sweep_scores.append(-1)

    log.info(f"M02: Best n_LV={best_lv}, val R²={best_r2:.4f}")
    plot_sweep(lv_range, sweep_scores, "n_LV (PLSR)", MODEL_NUM, save_dir)

    # Predict on individual test spectra
    Y_pred = softmax_normalize(best_pls.predict(F_test_s))

    metrics = compute_metrics(Y_test, Y_pred, F_train.shape[1])
    metrics["best_n_lv"] = best_lv
    metrics["nmf_k"] = C.NMF_K
    metrics["epochs"] = 0
    metrics["training_time"] = time.time() - t0

    log.info(f"M02: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    gc.collect()
    return Y_pred, metrics
