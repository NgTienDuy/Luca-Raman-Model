"""run_all_models.py v5 — runner"""
import os, sys, time, json, argparse, traceback
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from config import MODEL_NAMES, RESULTS_DIR, COMPARISON_DIR, LABEL_COLS, DATA_FILE
from utils import (load_data, aggregate_by_sample, split_data,
                   plot_summary_table,
                   evaluate, print_metrics, plot_comparison,
                   get_logger, load_results, INTERRUPTER)

log = get_logger("runner")


def _import(mid):
    import importlib
    return importlib.import_module({1:"model01",2:"model02",3:"model03",4:"model04",
                                    5:"model05",6:"model06",7:"model07",8:"model08",
                                    9:"model09",10:"model10"}[mid])


def run_model(mid, X_tr, X_te, Y_tr, Y_te, wns, retrain,
              X_te_raw=None, Y_te_raw=None, sid_te_raw=None):
    t0 = time.time()
    try:
        mod = _import(mid)
        kw  = dict(X_train=X_tr, X_test=X_te, Y_train=Y_tr, Y_test=Y_te, retrain=retrain,
                   X_test_raw=X_te_raw, Y_test_raw=Y_te_raw, sid_test_raw=sid_te_raw)
        if mid in (3,6,7,8,9,10):
            kw["wavenumbers"] = wns
        Y_pred, metrics = mod.run(**kw)
        elapsed = time.time()-t0
        return {"model_id":mid, "name":MODEL_NAMES[mid],
                "metrics":metrics, "elapsed_s":elapsed}
    except MemoryError:
        log.error(f"  ✗ Model {mid}: HET RAM"); return None
    except KeyboardInterrupt:
        log.warning(f"  Model {mid} bi gian doan"); raise
    except Exception as e:
        log.error(f"  ✗ Model {mid}: {e}")
        log.error(traceback.format_exc()); return None


def _print_table(all_res):
    for mid in sorted(all_res.keys()):
        r=all_res[mid]; m=r.get("metrics",{})
        r2=m.get("r2",float("nan")); mae=m.get("mae",float("nan"))
        rmse=m.get("rmse",float("nan")); t=r.get("elapsed_s",float("nan"))
        nm=r.get("name",MODEL_NAMES.get(mid,f"M{mid}"))
        print(f"  {mid:>3}  {nm:<35}  {r2:>8.4f}  {mae:>8.4f}  {rmse:>8.4f}  {t:>9.1f}s")
    valid={k:v for k,v in all_res.items()
           if not np.isnan(v.get("metrics",{}).get("r2",float("nan")))}
    if valid:
        best=max(valid,key=lambda k:valid[k]["metrics"]["r2"])
    print()
    import csv
    csv_path = os.path.join(COMPARISON_DIR, "comparison_table.csv")
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["model_id","name","r2","mae","rmse","mse","elapsed_s"])
        for mid in sorted(all_res.keys()):
            r=all_res[mid]; m=r.get("metrics",{})
            w.writerow([mid,r.get("name",""),m.get("r2",""),m.get("mae",""),
                        m.get("rmse",""),m.get("mse",""),r.get("elapsed_s","")])
    try: plot_summary_table(all_res, COMPARISON_DIR)
    except Exception as _e: log.warning(f"  Summary table: {_e}")


def run_chemistry_comparison(wns, X_tr, Y_tr, finished):
    try:
        from chemistry_report import (batch_chemistry, mean_profile, compare_models,
                                      format_report, save_chemistry_json, plot_bond_contribution)
        from utils import preprocess_batch
    except ImportError as e:
        log.warning(f"  chemistry_report import error: {e}"); return

    X_proc = preprocess_batch(X_tr, "full")
    profiles = batch_chemistry(X_proc, wns, Y_tr, LABEL_COLS)
    avg      = mean_profile(profiles)


    # Build per-model profiles
    model_profiles = {}
    for mid in [m for m in finished if m >= 6]:
        # Try to load model-specific chemistry from saved JSON
        import json as _json
        from config import get_model_dir as _gmd
        chem_path = os.path.join(_gmd(mid), f"model{mid:02d}_chemistry.json")
        if os.path.exists(chem_path):
            try:
                with open(chem_path) as _f: _cd = _json.load(_f)
                _prof_data = _cd.get("profile", {})
                from chemistry_report import ChemistryProfile, _comment_pH, _comment_polarity
                from chemistry_report import _comment_hydro, _comment_strength, _comment_arom
                from chemistry_report import _comment_amide, _comment_cryst
                _p = ChemistryProfile(**{k:v for k,v in _prof_data.items()
                                         if k in ChemistryProfile.__dataclass_fields__})
                _p.comments = {
                    "pH": _comment_pH(_p.pH_score), "polarity": _comment_polarity(_p.polarity),
                    "hydrophilicity": _comment_hydro(_p.hydrophilicity),
                    "bond_strength": _comment_strength(_p.bond_strength),
                    "aromaticity": _comment_arom(_p.aromaticity),
                    "amide_ratio": _comment_amide(_p.amide_I_III_ratio),
                    "crystallinity": _comment_cryst(_p.crystallinity_proxy),
                }
                model_profiles[f"M{mid}-{MODEL_NAMES[mid][:10]}"] = _p
            except Exception as _e:
                log.warning(f"  Load chemistry M{mid}: {_e}")
                model_profiles[f"M{mid}-{MODEL_NAMES[mid][:10]}"] = avg
        else:
            model_profiles[f"M{mid}-{MODEL_NAMES[mid][:10]}"] = avg

    if model_profiles:
        cmp_text = compare_models(model_profiles, save_path=COMPARISON_DIR)
        print(cmp_text)

    # Per-sample table
    log.info("\n  Per-sample physicochemical:")
    print(f"  {'Idx':<5} {'Est.pI':>6} {'pH':>7} {'Polar':>7} {'Hydro':>7} {'Arom':>7} {'Struct':<15} {'Dom.Bond'}")
    for i, prof in enumerate(profiles):
        print(f"  {i:<5} {prof.estimated_pI:>6.2f} {prof.pH_score:>+7.3f} "
              f"{prof.polarity:>7.3f} {prof.hydrophilicity:>7.3f} "
              f"{prof.aromaticity:>7.4f} {prof.secondary_structure[:14]:<15} {prof.dominant_bond}")

    # Save
    from dataclasses import asdict; import json
    out = {"avg_profile": asdict(avg),
           "per_sample": [{
               "idx": i, "estimated_pI": p.estimated_pI, "pH_score": p.pH_score,
               "polarity": p.polarity, "hydrophilicity": p.hydrophilicity,
               "bond_strength": p.bond_strength, "aromaticity": p.aromaticity,
               "amide_I_III_ratio": p.amide_I_III_ratio,
               "secondary_structure": p.secondary_structure,
               "aliphatic_index": p.aliphatic_index,
               "carboxylate_ratio": p.carboxylate_ratio,
               "amine_ratio": p.amine_ratio,
               "crystallinity_proxy": p.crystallinity_proxy,
               "spectral_entropy": p.spectral_entropy,
               "dominant_bond": p.dominant_bond,
               "top3_bonds": p.top3_bonds,
           } for i, p in enumerate(profiles)]}
    path = os.path.join(COMPARISON_DIR, "chemistry_all_models.json")
    with open(path,"w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x,"__float__") else str(x))

    # Comparison plot
    try:
        plot_bond_contribution(avg, "Train average",
                               os.path.join(COMPARISON_DIR, "chemistry_bond_contribution.png"))
        log.info(f"  Da luu chemistry files vao {COMPARISON_DIR}")
    except Exception as e:
        log.warning(f"  Chemistry plot loi: {e}")


def compare_only():
    all_res = {}
    for mid in range(1,11):
        r = load_results(mid)
        if r:
            r["name"] = r.get("name", MODEL_NAMES.get(mid, f"M{mid}"))
            all_res[mid] = r
    if not all_res:
        log.warning("Chua co ket qua."); return
    _print_table(all_res); plot_comparison(all_res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",           default=DATA_FILE)
    p.add_argument("--models",         default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--retrain",        action="store_true")
    p.add_argument("--compare-only",   action="store_true")
    p.add_argument("--chemistry-only", action="store_true")
    p.add_argument("--train-n",        type=int, default=48)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    if args.compare_only:
        compare_only(); return

    try:
        model_ids = [int(x.strip()) for x in args.models.split(",")]
    except ValueError:
        log.error("--models phai la so, vi du: 1,3,5"); sys.exit(1)

    if not os.path.exists(args.data):
        log.error(f"Khong tim thay: {args.data}"); sys.exit(1)

    log.info(f"Dang tai du lieu: {args.data}")
    X, Y, sids, wns = load_data(args.data, convert_wavelength=True)

    log.info("\nAggregate spectra theo sample...")
    X_agg, Y_agg, sid_agg = aggregate_by_sample(X, Y, sids, method="median")

    log.info(f"\nSplit: train_n={args.train_n}, seed={args.seed}")
    X_tr,X_te,Y_tr,Y_te,tr_set,te_set = split_data(
        X_agg, Y_agg, sid_agg, train_n=args.train_n, seed=args.seed)
    log.info(f"Train samples: {sorted(tr_set)}")
    log.info(f"Test  samples: {sorted(te_set)}")

    te_mask    = np.array([s in te_set for s in sids])
    X_te_raw   = X[te_mask]; Y_te_raw = Y[te_mask]; sid_te_raw = sids[te_mask]
    log.info(f"Raw test spectra: {X_te_raw.shape[0]} (6 samples x ~90 pho)")
    log.info(f"Raman shift: {wns.min():.1f}-{wns.max():.1f} cm-1\n")

    if args.chemistry_only:
        finished = [mid for mid in range(1,11) if load_results(mid) is not None]
        run_chemistry_comparison(wns, X_tr, Y_tr, finished)
        return

    all_res = {}; finished = []
    for mid in range(1,11):
        r = load_results(mid)
        if r:
            r["name"] = r.get("name", MODEL_NAMES.get(mid, f"M{mid}"))
            all_res[mid] = r

    t_total = time.time()
    for mid in model_ids:
        if INTERRUPTER.stop_requested:
            log.info("Runner dung theo yeu cau"); break
        res = run_model(mid, X_tr, X_te, Y_tr, Y_te, wns, args.retrain,
                        X_te_raw, Y_te_raw, sid_te_raw)
        if res:
            all_res[mid] = res; finished.append(mid)
            print(f"\n  Model {mid} ({MODEL_NAMES[mid]})")
            print_metrics(res["metrics"], indent=4)

    log.info(f"\nTong thoi gian: {time.time()-t_total:.1f}s")
    if all_res:
        _print_table(all_res)
        try: plot_comparison(all_res)
        except Exception as e: log.warning(f"Comparison plot loi: {e}")
        master = {mid: {"name":v.get("name",""), "metrics":v.get("metrics",{}),
                        "elapsed_s":v.get("elapsed_s",0)} for mid,v in all_res.items()}
        with open(os.path.join(COMPARISON_DIR,"all_models_summary.json"),"w") as f:
            json.dump(master, f, indent=2,
                      default=lambda x: float(x) if isinstance(x,(float,int)) else str(x))

    chem_done = [m for m in finished if m >= 6]
    if chem_done:
        try: run_chemistry_comparison(wns, X_tr, Y_tr, chem_done)
        except Exception as e:
            log.error(f"Chemistry loi: {e}"); log.error(traceback.format_exc())

    log.info("Hoan tat!")


if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except KeyboardInterrupt: print("\n[!] Dung."); sys.exit(0)
    except Exception as e:
        log.error(f"Loi: {e}"); log.error(traceback.format_exc()); sys.exit(1)
