"""
Chemistry report: extract physicochemical properties from spectra and composition.
"""
import numpy as np
import json, os, logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from spectral_knowledge import REGIONS, get_region_mask, extract_bond_features

log = logging.getLogger(__name__)

# Amino acid properties (approximate)
AA_MW = {"Alanine": 89.09, "Asparagine": 132.12, "Aspartic Acid": 133.10,
         "Glutamic Acid": 147.13, "Histidine": 155.16, "Glucosamine": 179.17}
AA_PI = {"Alanine": 6.00, "Asparagine": 5.41, "Aspartic Acid": 2.77,
         "Glutamic Acid": 3.22, "Histidine": 7.59, "Glucosamine": 6.91}
AA_POLARITY = {"Alanine": 0.3, "Asparagine": 0.7, "Aspartic Acid": 0.8,
               "Glutamic Acid": 0.8, "Histidine": 0.6, "Glucosamine": 0.7}
AA_HYDROPHILICITY = {"Alanine": 0.3, "Asparagine": 0.8, "Aspartic Acid": 0.9,
                     "Glutamic Acid": 0.9, "Histidine": 0.5, "Glucosamine": 0.8}


def extract_chemistry(spectrum, wavenumbers, composition=None):
    """Extract physicochemical properties from a single spectrum."""
    props = {}

    bf = extract_bond_features(spectrum, wavenumbers, REGIONS)
    # bf shape: (n_regions, 5) — [area, peak_height, peak_pos, FWHM, skew]

    # Spectral entropy
    s_pos = np.maximum(spectrum, 0)
    s_norm = s_pos / max(s_pos.sum(), 1e-10)
    s_norm = s_norm[s_norm > 0]
    props["spectral_entropy"] = float(-np.sum(s_norm * np.log(s_norm + 1e-15)))

    # Region-based ratios
    region_areas = bf[:, 0]

    # Amide I region (index 7: 1630-1700) vs Amide III region (index 3: 1100-1200)
    amide_i_area = region_areas[7] if len(region_areas) > 7 else 0
    amide_iii_area = region_areas[3] if len(region_areas) > 3 else 0
    props["amide_I_III_ratio"] = float(amide_i_area / max(amide_iii_area, 1e-10))

    # Amide I position (peak position in amide I region)
    props["amide_I_position"] = float(bf[7, 2]) if len(bf) > 7 else 0

    # Carboxylate ratio: C=O(carboxyl) area / C-O(carboxyl) area
    co_double = region_areas[8] if len(region_areas) > 8 else 0  # 1700-1800
    co_single = region_areas[4] if len(region_areas) > 4 else 0  # 1200-1350
    props["carboxylate_ratio"] = float(co_double / max(co_single, 1e-10))

    # Amine ratio: N-H region area / total area
    nh_area = region_areas[6] if len(region_areas) > 6 else 0  # 1500-1630
    total_area = max(region_areas.sum(), 1e-10)
    props["amine_ratio"] = float(nh_area / total_area)

    # CN/CC ratio
    cn_area = region_areas[3] if len(region_areas) > 3 else 0  # 1100-1200 (C-N amide)
    cc_area = region_areas[2] if len(region_areas) > 2 else 0  # 800-1100 (C-C)
    props["CN_CC_ratio"] = float(cn_area / max(cc_area, 1e-10))

    # Crystallinity proxy (peak sharpness in C-C region)
    cc_fwhm = bf[2, 3] if len(bf) > 2 else 0
    props["crystallinity_proxy"] = float(1.0 / max(cc_fwhm, 1e-10))

    # Aromaticity: C=C region area / total
    cc_aromatic = region_areas[6] if len(region_areas) > 6 else 0
    props["aromaticity"] = float(cc_aromatic / total_area)

    # Dominant bond region
    dom_idx = np.argmax(region_areas)
    props["dominant_region"] = REGIONS[dom_idx].name if dom_idx < len(REGIONS) else "unknown"

    # Top 3 bond regions by area
    top3_idx = np.argsort(region_areas)[::-1][:3]
    props["top3_regions"] = [REGIONS[i].name for i in top3_idx if i < len(REGIONS)]

    # Composition-derived properties
    if composition is not None and len(composition) == 6:
        comp = np.array(composition)
        names = C.AA_NAMES

        # pH score (weighted pI)
        pi_vals = np.array([AA_PI[n] for n in names])
        props["pH_score"] = float(np.sum(comp * pi_vals))

        # Polarity
        pol_vals = np.array([AA_POLARITY[n] for n in names])
        props["polarity"] = float(np.sum(comp * pol_vals))

        # Hydrophilicity
        hyd_vals = np.array([AA_HYDROPHILICITY[n] for n in names])
        props["hydrophilicity"] = float(np.sum(comp * hyd_vals))

        # Average MW
        mw_vals = np.array([AA_MW[n] for n in names])
        props["estimated_MW"] = float(np.sum(comp * mw_vals))

        # Estimated pI
        props["estimated_pI"] = float(np.sum(comp * pi_vals))

        # Bond strength (from composition-weighted region areas)
        props["bond_strength"] = float(np.mean(region_areas))
    else:
        # Spectral-only estimates
        props["pH_score"] = float(props["carboxylate_ratio"] * 3 + props["amine_ratio"] * 5)
        props["polarity"] = float(props["amine_ratio"] + props["carboxylate_ratio"])
        props["hydrophilicity"] = float(props["amine_ratio"] * 0.6 + props["carboxylate_ratio"] * 0.4)
        props["bond_strength"] = float(np.mean(region_areas))

    return props


def extract_chemistry_batch(X, wavenumbers, compositions=None):
    """Extract chemistry for a batch of spectra."""
    results = []
    for i in range(len(X)):
        comp = compositions[i] if compositions is not None else None
        results.append(extract_chemistry(X[i], wavenumbers, comp))
    return results


def compare_models_table(model_chems, save_dir):
    """
    Create a simple comparison table of chemistry across models 6-10.
    model_chems: dict of {model_num: chemistry_dict}
    """
    if not model_chems:
        log.warning("No chemistry data to compare")
        return

    keys = ["pH_score", "polarity", "hydrophilicity", "bond_strength",
            "aromaticity", "amide_I_III_ratio", "carboxylate_ratio",
            "amine_ratio", "CN_CC_ratio", "spectral_entropy"]

    models = sorted(model_chems.keys())
    n_models = len(models)
    n_keys = len(keys)

    fig, ax = plt.subplots(figsize=(12, max(5, n_keys * 0.4 + 2)))
    ax.axis('off')

    col_labels = [f"M{m}" for m in models] + ["Consensus"]
    row_labels = keys

    cell_text = []
    for k in keys:
        row = []
        vals = []
        for m in models:
            v = model_chems[m].get(k, 0)
            if isinstance(v, (int, float)):
                row.append(f"{v:.4f}")
                vals.append(v)
            else:
                row.append(str(v))
        # Consensus = median
        if vals:
            row.append(f"{np.median(vals):.4f}")
        else:
            row.append("—")
        cell_text.append(row)

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=col_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title("Chemistry Comparison (M6–M10)", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    out = os.path.join(save_dir, "chemistry_comparison.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")

    # JSON
    out_json = os.path.join(save_dir, "chemistry_comparison.json")
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in model_chems.items()}, f, indent=2, default=str)
