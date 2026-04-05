"""
Shared helpers for models M6–M10: bond detection plots, chemistry extraction.
"""
import os, json, logging, gc
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from spectral_knowledge import REGIONS, BOND_TYPES, get_region_mask, extract_bond_features
from chemistry_report import extract_chemistry
from utils import snip_baseline, area_normalization

log = logging.getLogger(__name__)


def _clean_spectrum(spectrum):
    """Ensure spectrum is baseline-corrected and non-negative for display."""
    s = spectrum.copy()
    s = snip_baseline(s, max_iter=30)
    s = np.maximum(s, 0)
    return s


def _find_bond_at_wavenumber(wn):
    """Identify which bond types could produce a peak at this wavenumber."""
    matches = []
    for reg in REGIONS:
        if reg.wn_min <= wn < reg.wn_max and reg.bond_types_contained:
            for bt_id in reg.bond_types_contained:
                matches.append(BOND_TYPES[bt_id]["name"])
    return matches


def generate_bond_detection_plots(X_test, wavenumbers, sample_ids, model_num, save_dir):
    """Generate bond detection plots: find peaks, label with bond names."""
    unique_sids = np.unique(sample_ids)
    for sid in unique_sids:
        mask = sample_ids == sid
        median_spec = np.median(X_test[mask], axis=0)

        # Clean spectrum: baseline correct, ensure positive
        spec_clean = _clean_spectrum(median_spec)
        # Smooth for peak finding
        spec_smooth = savgol_filter(spec_clean, 11, 3)
        spec_smooth = np.maximum(spec_smooth, 0)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(wavenumbers, spec_smooth, 'k-', linewidth=1.0)
        ax.fill_between(wavenumbers, 0, spec_smooth, alpha=0.08, color='steelblue')

        # Find peaks
        prominence = np.max(spec_smooth) * 0.05
        peaks, props = find_peaks(spec_smooth, prominence=prominence,
                                   distance=15, height=np.max(spec_smooth) * 0.03)

        if len(peaks) > 0:
            # Sort by height, keep top 15
            heights = spec_smooth[peaks]
            top_idx = np.argsort(heights)[::-1][:15]
            peaks = peaks[top_idx]

            # Color cycle
            cmap = plt.cm.Set1(np.linspace(0, 1, 10))

            used_y = []  # track label positions to avoid overlap
            for i, pk in enumerate(peaks):
                wn_pk = wavenumbers[pk]
                h_pk = spec_smooth[pk]

                # Find bond assignment
                bonds = _find_bond_at_wavenumber(wn_pk)
                if bonds:
                    label = bonds[0]  # primary bond
                    short = label.replace(" (", "\n(")
                else:
                    label = "—"
                    short = "—"

                color = cmap[i % len(cmap)]
                ax.plot(wn_pk, h_pk, 'v', color=color, markersize=8, zorder=5)

                # Smart label placement: alternate up/down
                offset = h_pk + np.max(spec_smooth) * 0.06
                # Avoid overlap
                for uy in used_y:
                    if abs(offset - uy) < np.max(spec_smooth) * 0.08 and abs(wn_pk - uy) < 80:
                        offset += np.max(spec_smooth) * 0.08
                used_y.append(offset)

                ax.annotate(short, xy=(wn_pk, h_pk),
                            xytext=(wn_pk, offset),
                            fontsize=6, ha='center', color=color,
                            arrowprops=dict(arrowstyle='-', color=color, lw=0.5),
                            fontweight='bold')

        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Intensity (a.u.)", fontsize=11)
        ax.set_title(f"M{model_num:02d} — Bond Detection: {sid}", fontsize=13)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        out = os.path.join(save_dir, f"model{model_num:02d}_bonddetect_{sid}.png")
        fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
        plt.close(fig)

    log.info(f"  Saved bond detection plots for {len(unique_sids)} samples")


def generate_chemistry_bonds_plot(X_test, wavenumbers, sample_ids, model_num, save_dir):
    """Generate chemistry bonds summary: positive areas from baseline-corrected spectra."""
    unique_sids = np.unique(sample_ids)
    region_names = [r.name[:25] for r in REGIONS]

    all_areas = []
    for sid in unique_sids:
        mask = sample_ids == sid
        med_spec = np.median(X_test[mask], axis=0)
        # Clean before feature extraction
        spec_clean = _clean_spectrum(med_spec)
        bf = extract_bond_features(spec_clean, wavenumbers, REGIONS)
        all_areas.append(np.abs(bf[:, 0]))  # absolute areas

    all_areas = np.array(all_areas)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(region_names))
    width = 0.12
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, sid in enumerate(unique_sids):
        ax.bar(x + i * width, all_areas[i], width,
               label=str(sid), alpha=0.85, color=colors[i % len(colors)])

    ax.set_xlabel("Bond Region", fontsize=11)
    ax.set_ylabel("Area (a.u.)", fontsize=11)
    ax.set_title(f"M{model_num:02d} — Bond Region Areas by Sample", fontsize=13)
    ax.set_xticks(x + width * len(unique_sids) / 2)
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(save_dir, f"model{model_num:02d}_chemistry_bonds.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")


def generate_chemistry_json(X_test, wavenumbers, Y_pred, sample_ids, model_num, save_dir):
    """Generate chemistry.json from test predictions."""
    unique_sids = np.unique(sample_ids)
    chem_results = {}

    for sid in unique_sids:
        mask = sample_ids == sid
        med_spec = np.median(X_test[mask], axis=0)
        spec_clean = _clean_spectrum(med_spec)
        med_comp = np.median(Y_pred[mask], axis=0)
        props = extract_chemistry(spec_clean, wavenumbers, med_comp)
        chem_results[str(sid)] = props

    avg_chem = {}
    numeric_keys = [k for k in list(chem_results.values())[0].keys()
                    if isinstance(list(chem_results.values())[0][k], (int, float))]
    for k in numeric_keys:
        vals = [chem_results[s][k] for s in chem_results
                if isinstance(chem_results[s].get(k), (int, float))]
        avg_chem[k] = float(np.mean(vals)) if vals else 0.0

    out = os.path.join(save_dir, f"model{model_num:02d}_chemistry.json")
    with open(out, 'w') as f:
        json.dump({"per_sample": chem_results, "average": avg_chem}, f, indent=2, default=str)
    log.info(f"  Saved {out}")
    return avg_chem


def post_process_advanced_model(X_test, wavenumbers, Y_pred, sample_ids, model_num, save_dir):
    """Full post-processing for M6-M10."""
    generate_bond_detection_plots(X_test, wavenumbers, sample_ids, model_num, save_dir)
    generate_chemistry_bonds_plot(X_test, wavenumbers, sample_ids, model_num, save_dir)
    avg_chem = generate_chemistry_json(X_test, wavenumbers, Y_pred, sample_ids, model_num, save_dir)
    gc.collect()
    return avg_chem
