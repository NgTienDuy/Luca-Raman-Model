"""
Model 01: Ridge Regression with Softmax Output — Linear baseline.
Train on sample-level medians. Predict on individual test spectra.
"""
import os, time, logging, gc
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import config as C
from utils import (preprocess_batch, compute_metrics, aggregate_by_sample,
                   plot_scatter_aggregated, plot_scatter_raw,
                   plot_sweep, save_results, set_seeds)

log = logging.getLogger(__name__)
MODEL_NUM = 1


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)

    sid_train = kw.get("sid_train")
    sid_val = kw.get("sid_val")
    sid_test = kw.get("sid_test")

    # Aggregate to sample medians for training
    Xtr_m, Ytr_m, _ = aggregate_by_sample(X_train, Y_train, sid_train)
    Xva_m, Yva_m, _ = aggregate_by_sample(X_val, Y_val, sid_val)

    Xtr_m = preprocess_batch(Xtr_m, 'snv')
    Xva_m = preprocess_batch(Xva_m, 'snv')

    # Log-transform targets and mean-center
    Ytr_log = np.log(np.clip(Ytr_m, 1e-6, None))
    Ytr_log -= Ytr_log.mean(axis=1, keepdims=True)

    # Sweep alpha
    log.info("M01: Sweeping alphas...")
    alphas = C.RIDGE_ALPHAS
    sweep_scores = []
    best_r2 = -1e9
    best_model = None
    best_alpha = None

    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr").fit(Xtr_m, Ytr_log)
        pred = softmax(model.predict(Xva_m), axis=1)
        r2 = r2_score(Yva_m, pred, multioutput="uniform_average")
        sweep_scores.append(r2)
        log.info(f"  alpha={alpha:>8} -> val R²={r2:.4f}")
        if r2 > best_r2:
            best_r2, best_model, best_alpha = r2, model, alpha

    log.info(f"M01: Best alpha={best_alpha}, val R²={best_r2:.4f}")
    plot_sweep(alphas, sweep_scores, "Alpha (Ridge)", MODEL_NUM, save_dir)

    # Predict on ALL individual test spectra
    Xt_proc = preprocess_batch(X_test, 'snv')
    Y_pred = softmax(best_model.predict(Xt_proc), axis=1)

    metrics = compute_metrics(Y_test, Y_pred)
    metrics["best_alpha"] = best_alpha
    metrics["epochs"] = len(alphas)
    metrics["training_time"] = time.time() - t0

    log.info(f"M01: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    gc.collect()
    return Y_pred, metrics
