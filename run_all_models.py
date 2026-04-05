"""
CLI runner: load data → split → run models → comparison.
Usage:
  python run_all_models.py --data data.csv
  python run_all_models.py --data data.csv --models 1,4,7
  python run_all_models.py --data data.csv --retrain
  python run_all_models.py --compare-only
"""
import os, sys, json, time, logging, argparse, gc, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from utils import load_data, split_data, set_seeds

log = logging.getLogger("runner")


def run_model(num, X_train, X_val, X_test, Y_train, Y_val, Y_test,
              wavenumbers, retrain, kw):
    """Import and run a single model."""
    mod = __import__(f"model{num:02d}")
    return mod.run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
                   wavenumbers=wavenumbers, retrain=retrain, **kw)


def generate_comparison(all_results):
    """Generate all comparison outputs."""
    comp_dir = os.path.join(C.RESULTS_DIR, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # ── comparison_table.csv ────────────────────────────────────────────
    import csv
    headers = ["Model", "Name", "R2", "Adj_R2", "MAE", "RMSE", "MAPE%",
               "MaxErr", "Epochs", "Time(s)"]
    rows = []
    for m_num in sorted(all_results.keys()):
        met = all_results[m_num]
        rows.append([
            f"M{m_num:02d}",
            C.MODEL_NAMES.get(m_num, ""),
            f"{met.get('R2', 0):.4f}",
            f"{met.get('Adj_R2', 0):.4f}",
            f"{met.get('MAE', 0):.4f}",
            f"{met.get('RMSE', 0):.4f}",
            f"{met.get('MAPE', 0):.1f}",
            f"{met.get('MaxError', 0):.4f}",
            str(met.get('epochs', 0)),
            f"{met.get('training_time', 0):.1f}",
        ])

    csv_path = os.path.join(comp_dir, "comparison_table.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    log.info(f"  Saved {csv_path}")

    # ── comparison.png (horizontal bar chart) ───────────────────────────
    models = sorted(all_results.keys())
    r2_vals = [all_results[m].get('R2', 0) for m in models]
    mae_vals = [all_results[m].get('MAE', 0) for m in models]
    names = [f"M{m:02d}" for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(models) * 0.5 + 1)))

    y_pos = np.arange(len(models))
    colors_r2 = ['#2ecc71' if v > 0.4 else '#f39c12' if v >= 0 else '#e74c3c' for v in r2_vals]
    ax1.barh(y_pos, r2_vals, color=colors_r2, edgecolor='k', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("R²")
    ax1.set_title("R² (higher = better)")
    ax1.axvline(0, color='k', linewidth=0.5)
    for i, v in enumerate(r2_vals):
        ax1.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)

    colors_mae = ['#2ecc71' if v < 0.1 else '#f39c12' if v < 0.2 else '#e74c3c' for v in mae_vals]
    ax2.barh(y_pos, mae_vals, color=colors_mae, edgecolor='k', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("MAE")
    ax2.set_title("MAE (lower = better)")
    for i, v in enumerate(mae_vals):
        ax2.text(v + 0.002, i, f"{v:.4f}", va='center', fontsize=9)

    fig.suptitle("Model Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(comp_dir, "comparison.png"), dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved comparison.png")

    # ── models_summary_table.png ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, max(4, len(models) * 0.45 + 2)))
    ax.axis('off')

    cell_text = []
    best_r2_idx = np.argmax(r2_vals)
    for i, row in enumerate(rows):
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Header styling
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best R²
    if len(rows) > 0:
        for j in range(len(headers)):
            table[best_r2_idx + 1, j].set_facecolor('#d5f5e3')

    ax.set_title("All Models — Summary", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(comp_dir, "models_summary_table.png"),
                dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved models_summary_table.png")

    # ── all_models_summary.json ─────────────────────────────────────────
    summary = {}
    for m_num in sorted(all_results.keys()):
        met = all_results[m_num]
        summary[f"M{m_num:02d}"] = {
            "name": C.MODEL_NAMES.get(m_num, ""),
            "R2": met.get('R2', 0),
            "Adj_R2": met.get('Adj_R2', 0),
            "MAE": met.get('MAE', 0),
            "RMSE": met.get('RMSE', 0),
            "MAPE": met.get('MAPE', 0),
            "MaxError": met.get('MaxError', 0),
            "epochs": met.get('epochs', 0),
            "training_time": met.get('training_time', 0),
            "per_aa": met.get('per_aa', {}),
        }
    json_path = os.path.join(comp_dir, "all_models_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Saved {json_path}")

    # ── chemistry_comparison.png (M6–M10) ──────────────────────────────
    chem_data = {}
    for m_num in sorted(all_results.keys()):
        if m_num >= 6:
            # Try loading from chemistry.json files
            chem_path = os.path.join(C.RESULTS_DIR, f"model{m_num:02d}",
                                      f"model{m_num:02d}_chemistry.json")
            if os.path.exists(chem_path):
                with open(chem_path) as f:
                    cdata = json.load(f)
                chem_data[m_num] = cdata.get("average", {})
            elif 'chemistry' in all_results[m_num]:
                chem_data[m_num] = all_results[m_num]['chemistry']

    if chem_data:
        from chemistry_report import compare_models_table
        compare_models_table(chem_data, comp_dir)

    log.info("  Comparison generation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run Raman AA models")
    parser.add_argument("--data", type=str, default=C.DATA_PATH)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model numbers (e.g., 1,4,7)")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    set_seeds()

    # Create directories
    for d in [C.RESULTS_DIR, C.CHECKPOINTS_DIR, C.PREDICTIONS_DIR, C.ANALYSIS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Generate bond region report
    log.info("Generating bond region report...")
    from bond_region_report import generate_report
    generate_report(os.path.join(C.ANALYSIS_DIR, "bond_region_report.txt"))

    # Determine which models to run
    if args.models:
        model_nums = [int(x.strip()) for x in args.models.split(",")]
    else:
        model_nums = list(range(1, 11))

    if args.compare_only:
        # Load existing results
        all_results = {}
        for m in range(1, 11):
            rpath = os.path.join(C.RESULTS_DIR, f"model{m:02d}", f"model{m:02d}_results.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    all_results[m] = json.load(f)
        if all_results:
            generate_comparison(all_results)
        else:
            log.warning("No results found for comparison.")
        return

    # Load and split data
    log.info("=" * 60)
    log.info("RAMAN AMINO ACID MIXTURE ANALYSIS")
    log.info("=" * 60)
    X, Y, wavenumbers, sample_ids = load_data(args.data)
    Xtr, Xv, Xt, Ytr, Yv, Yt, sid_tr, sid_v, sid_t = split_data(X, Y, sample_ids)

    kw = {"sid_train": sid_tr, "sid_val": sid_v, "sid_test": sid_t}

    all_results = {}
    for m_num in model_nums:
        log.info("=" * 60)
        log.info(f"MODEL {m_num:02d}: {C.MODEL_NAMES.get(m_num, '')}")
        log.info("=" * 60)
        try:
            Y_pred, metrics = run_model(m_num, Xtr, Xv, Xt, Ytr, Yv, Yt,
                                        wavenumbers, args.retrain, kw)
            all_results[m_num] = metrics

            # Save predictions
            pred_path = os.path.join(C.PREDICTIONS_DIR, f"model{m_num:02d}_predictions.npy")
            np.save(pred_path, Y_pred)

        except MemoryError:
            log.error(f"  MemoryError in model {m_num:02d} — skipping")
            gc.collect()
        except KeyboardInterrupt:
            log.warning(f"  KeyboardInterrupt at model {m_num:02d} — saving and continuing")
            gc.collect()
        except Exception as e:
            log.error(f"  Error in model {m_num:02d}: {e}")
            traceback.print_exc()
            gc.collect()

    # Also load any previously saved results for models we didn't run
    for m in range(1, 11):
        if m not in all_results:
            rpath = os.path.join(C.RESULTS_DIR, f"model{m:02d}", f"model{m:02d}_results.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    all_results[m] = json.load(f)

    # Generate comparison
    if all_results:
        log.info("=" * 60)
        log.info("GENERATING COMPARISON")
        log.info("=" * 60)
        generate_comparison(all_results)

    log.info("=" * 60)
    log.info("ALL DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
