"""
run_random_split.py — Run all 10 models with traditional 60/20/20 random split.
Compares against group-stratified split to show generalization gap.

Output: results_random_split/
  - comparison.png, comparison_table.png
  - model01_scatter_raw.png … model10_scatter_raw.png
  - model01_loss.png … model10_loss.png
"""
import os, sys, json, time, logging, gc, traceback, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Setup ───────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
OUT_DIR = "results_random_split"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("random_split")

import config as C
from utils import load_data, set_seeds, compute_metrics, softmax_normalize, plot_scatter_raw

# ── Monkey-patch to disable bond/chemistry post-processing ─────────────
import model_helpers
def _noop_post_process(*a, **kw):
    return {}
model_helpers.post_process_advanced_model = _noop_post_process


def random_split(X, Y, sample_ids, seed=42):
    """Traditional 60/20/20 random split (no grouping by sample)."""
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    log.info(f"  Random split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return (X[train_idx], X[val_idx], X[test_idx],
            Y[train_idx], Y[val_idx], Y[test_idx],
            sample_ids[train_idx], sample_ids[val_idx], sample_ids[test_idx])


def run_model(num, Xtr, Xv, Xt, Ytr, Yv, Yt, wavenumbers, kw):
    """Import and run a single model, redirect outputs."""
    orig_results = C.RESULTS_DIR
    C.RESULTS_DIR = OUT_DIR

    sub = os.path.join(OUT_DIR, f"model{num:02d}")
    os.makedirs(sub, exist_ok=True)

    mod = __import__(f"model{num:02d}")
    Y_pred, metrics = mod.run(Xtr, Xv, Xt, Ytr, Yv, Yt,
                              wavenumbers=wavenumbers, retrain=True, **kw)

    C.RESULTS_DIR = orig_results
    return Y_pred, metrics


def collect_outputs():
    """Move scatter_raw and loss/sweep PNGs to OUT_DIR root."""
    for m in range(1, 11):
        sub = os.path.join(OUT_DIR, f"model{m:02d}")
        if not os.path.isdir(sub):
            continue

        raw = os.path.join(sub, f"model{m:02d}_scatter_raw.png")
        if os.path.exists(raw):
            shutil.copy2(raw, os.path.join(OUT_DIR, f"model{m:02d}_scatter_raw.png"))

        loss = os.path.join(sub, f"model{m:02d}_loss.png")
        sweep = os.path.join(sub, f"model{m:02d}_sweep.png")
        if os.path.exists(loss):
            shutil.copy2(loss, os.path.join(OUT_DIR, f"model{m:02d}_loss.png"))
        elif os.path.exists(sweep):
            shutil.copy2(sweep, os.path.join(OUT_DIR, f"model{m:02d}_loss.png"))

        rjson = os.path.join(sub, f"model{m:02d}_results.json")
        if os.path.exists(rjson):
            shutil.copy2(rjson, os.path.join(OUT_DIR, f"model{m:02d}_results.json"))

        shutil.rmtree(sub)


def generate_comparison(all_results):
    """Generate comparison bar chart + table."""
    models = sorted(all_results.keys())
    r2_vals = [all_results[m].get('R2', 0) for m in models]
    mae_vals = [all_results[m].get('MAE', 0) for m in models]
    names = [f"M{m:02d}" for m in models]

    # ── comparison.png ──────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    y_pos = np.arange(len(models))

    colors_r2 = ['#2ecc71' if v > 0.4 else '#f39c12' if v >= 0 else '#e74c3c' for v in r2_vals]
    ax1.barh(y_pos, r2_vals, color=colors_r2, edgecolor='k', linewidth=0.5)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(names)
    ax1.set_xlabel("R²"); ax1.set_title("R² (higher = better)")
    ax1.axvline(0, color='k', linewidth=0.5)
    for i, v in enumerate(r2_vals):
        ax1.text(max(v + 0.01, 0.02), i, f"{v:.3f}", va='center', fontsize=9)

    colors_mae = ['#2ecc71' if v < 0.05 else '#f39c12' if v < 0.1 else '#e74c3c' for v in mae_vals]
    ax2.barh(y_pos, mae_vals, color=colors_mae, edgecolor='k', linewidth=0.5)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(names)
    ax2.set_xlabel("MAE"); ax2.set_title("MAE (lower = better)")
    for i, v in enumerate(mae_vals):
        ax2.text(v + 0.001, i, f"{v:.4f}", va='center', fontsize=9)

    fig.suptitle("Model Comparison — Random 60/20/20 Split", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── comparison_table.png ────────────────────────────────────────────
    headers = ["Model", "Name", "R²", "Adj R²", "MAE", "RMSE", "MAPE%", "MaxErr", "Epochs", "Time(s)"]
    rows = []
    for m in models:
        met = all_results[m]
        rows.append([
            f"M{m:02d}", C.MODEL_NAMES.get(m, ""),
            f"{met.get('R2', 0):.4f}", f"{met.get('Adj_R2', 0):.4f}",
            f"{met.get('MAE', 0):.4f}", f"{met.get('RMSE', 0):.4f}",
            f"{met.get('MAPE', 0):.1f}", f"{met.get('MaxError', 0):.4f}",
            str(met.get('epochs', 0)), f"{met.get('training_time', 0):.1f}",
        ])

    best_idx = int(np.argmax(r2_vals))

    fig, ax = plt.subplots(figsize=(16, max(4, len(models) * 0.45 + 2)))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor('#d5f5e3')

    ax.set_title("All Models — Random Split Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparison_table.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved comparison.png and comparison_table.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model numbers")
    args = parser.parse_args()

    if args.models:
        model_nums = [int(x) for x in args.models.split(",")]
    else:
        model_nums = list(range(1, 11))

    set_seeds()
    log.info("=" * 60)
    log.info("RANDOM SPLIT BENCHMARK — 60/20/20")
    log.info("=" * 60)

    X, Y, wavenumbers, sample_ids = load_data(C.DATA_PATH)
    Xtr, Xv, Xt, Ytr, Yv, Yt, sid_tr, sid_v, sid_t = random_split(X, Y, sample_ids)
    del X, Y; gc.collect()

    kw = {"sid_train": sid_tr, "sid_val": sid_v, "sid_test": sid_t}

    os.makedirs(C.CHECKPOINTS_DIR, exist_ok=True)
    for f in os.listdir(C.CHECKPOINTS_DIR):
        os.remove(os.path.join(C.CHECKPOINTS_DIR, f))

    all_results = {}
    for m_num in model_nums:
        log.info("=" * 60)
        log.info(f"MODEL {m_num:02d}: {C.MODEL_NAMES.get(m_num, '')}")
        log.info("=" * 60)
        try:
            Y_pred, metrics = run_model(m_num, Xtr, Xv, Xt, Ytr, Yv, Yt, wavenumbers, kw)
            all_results[m_num] = metrics
            log.info(f"  -> R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
        except MemoryError:
            log.error(f"  MemoryError in M{m_num:02d} — skipping")
            gc.collect()
        except Exception as e:
            log.error(f"  Error in M{m_num:02d}: {e}")
            traceback.print_exc()
            gc.collect()

    collect_outputs()

    # Load any previously saved results
    for m in range(1, 11):
        if m not in all_results:
            rpath = os.path.join(OUT_DIR, f"model{m:02d}",  f"model{m:02d}_results.json")
            # Also try root dir (after collect)
            rpath2 = os.path.join(OUT_DIR, f"model{m:02d}_results.json")
            for rp in [rpath, rpath2]:
                if os.path.exists(rp):
                    with open(rp) as f:
                        all_results[m] = json.load(f)
                    break

    if all_results:
        generate_comparison(all_results)

    pngs = [f for f in os.listdir(OUT_DIR) if f.endswith('.png')]
    log.info(f"\nOutput: {len(pngs)} PNG files in {OUT_DIR}/")
    for f in sorted(pngs):
        log.info(f"  {f}")

    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
