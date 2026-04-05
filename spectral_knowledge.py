"""
Spectral knowledge: 17 bond types, region mapping, feature extraction.
Regions partition 267–2004 cm⁻¹ with no gaps. Each region documents which
of the 17 bond types it contains.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ═══════════════════════════════════════════════════════════════════════════
#  17 BOND TYPES
# ═══════════════════════════════════════════════════════════════════════════

BOND_TYPES = {
    1:  {"name": "C-C (aliphatic)",         "range": (800, 1100),   "present_in": "All 6"},
    2:  {"name": "C=C (aromatic)",           "range": (1500, 1600),  "present_in": "Histidine"},
    3:  {"name": "C-H (sp3)",               "range": (1350, 1500),  "present_in": "All 6",
         "note": "Overtone/deformation of 2800-3000"},
    4:  {"name": "C-H (sp2)",               "range": (1000, 1100),  "present_in": "Histidine",
         "note": "Overtone of 3000-3100"},
    5:  {"name": "C-N (amine)",             "range": (1000, 1150),  "present_in": "All 6"},
    6:  {"name": "C-N (amide)",             "range": (1100, 1250),  "present_in": "Asparagine"},
    7:  {"name": "C=N (imine/aromatic)",    "range": (1550, 1650),  "present_in": "Histidine"},
    8:  {"name": "C=O (carboxyl)",          "range": (1680, 1750),  "present_in": "All 6"},
    9:  {"name": "C=O (amide)",             "range": (1630, 1690),  "present_in": "Asparagine"},
    10: {"name": "C-O (carboxyl)",          "range": (1200, 1350),  "present_in": "All 6"},
    11: {"name": "C-O (alcohol)",           "range": (1000, 1100),  "present_in": "Glucosamine"},
    12: {"name": "C-O (ether)",             "range": (1070, 1150),  "present_in": "Glucosamine"},
    13: {"name": "N-H (primary amine)",     "range": (1550, 1650),  "present_in": "All 6",
         "note": "Bending mode; stretch 3300-3500 outside range"},
    14: {"name": "N-H (primary amide)",     "range": (1600, 1680),  "present_in": "Asparagine"},
    15: {"name": "N-H (secondary aromatic)","range": (1550, 1630),  "present_in": "Histidine",
         "note": "Overtone/combination of 3100-3250"},
    16: {"name": "O-H (carboxylic)",        "range": (1300, 1450),  "present_in": "All 6",
         "note": "Bending/combination of broad 2500-3300"},
    17: {"name": "O-H (alcoholic)",         "range": (1200, 1400),  "present_in": "Glucosamine",
         "note": "Bending mode of 3200-3600"},
}

# Mapping: bond type → amino acids
BOND_TO_AA = {
    1:  ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    2:  ["Histidine"],
    3:  ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    4:  ["Histidine"],
    5:  ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    6:  ["Asparagine"],
    7:  ["Histidine"],
    8:  ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    9:  ["Asparagine"],
    10: ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    11: ["Glucosamine"],
    12: ["Glucosamine"],
    13: ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    14: ["Asparagine"],
    15: ["Histidine"],
    16: ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid", "Histidine", "Glucosamine"],
    17: ["Glucosamine"],
}


# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL REGIONS (partition 267–2004 cm⁻¹ with no gaps)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BondRegion:
    name: str
    wn_min: float
    wn_max: float
    bond_types_contained: List[int]  # indices into BOND_TYPES
    polarity: str
    bond_strength: str
    description: str


REGIONS = [
    BondRegion("Low-frequency lattice/skeletal",
               267, 500, [],
               "low", "weak",
               "Lattice modes, skeletal deformations, torsions. No specific bond type from the 17."),
    BondRegion("Skeletal C-C/C-N stretch (low)",
               500, 800, [],
               "moderate", "moderate",
               "C-C and C-N skeletal stretches, ring breathing (low end). Transitional region."),
    BondRegion("C-C aliphatic + C-N amine + C-O alcohol/ether + C-H sp2 overtone",
               800, 1100, [1, 4, 5, 11, 12],
               "moderate", "moderate-strong",
               "Major region: C-C(sp3) stretch #1, C-N(amine) #5, C-O(alcohol) #11, "
               "C-O(ether) #12, C-H(sp2) overtone #4. Ring breathing modes."),
    BondRegion("C-N amide stretch",
               1100, 1200, [6],
               "moderate-high", "moderate",
               "C-N(amide) stretch #6 from asparagine side chain."),
    BondRegion("C-O carboxyl + O-H bending",
               1200, 1350, [10, 17],
               "high", "moderate-strong",
               "C-O(carboxyl) single bond #10. O-H(alcoholic) bending #17."),
    BondRegion("C-H sp3 deformation + O-H carboxylic bending",
               1350, 1500, [3, 16],
               "moderate", "moderate",
               "C-H(sp3) bending/deformation #3. O-H(carboxylic) bending #16."),
    BondRegion("C=C aromatic + C=N + N-H bends",
               1500, 1630, [2, 7, 13, 15],
               "high", "strong",
               "C=C(aromatic) #2, C=N(imine) #7, N-H(amine) bend #13, "
               "N-H(secondary aromatic) #15."),
    BondRegion("Amide I/II + N-H amide bend",
               1630, 1700, [9, 14],
               "high", "strong",
               "C=O(amide) #9, N-H(primary amide) bend #14. Amide I region."),
    BondRegion("C=O carboxyl stretch",
               1700, 1800, [8],
               "high", "strong",
               "C=O(carboxyl) stretch #8 from -COOH."),
    BondRegion("High-frequency overtone/combination",
               1800, 2004, [],
               "low", "weak",
               "Overtone and combination bands. No primary fundamentals from 17 types in this window."),
]

N_REGIONS = len(REGIONS)


def get_region_mask(wavenumbers, region):
    """Boolean mask for wavenumbers within a region."""
    return (wavenumbers >= region.wn_min) & (wavenumbers < region.wn_max)


def extract_bond_features(spectrum, wavenumbers, regions=None):
    """
    Extract features per bond region:
      [area, peak_height, peak_position, FWHM, skewness] per region.
    Returns shape: (n_regions, 5)
    """
    if regions is None:
        regions = REGIONS
    features = np.zeros((len(regions), 5), dtype=np.float32)

    for r, reg in enumerate(regions):
        mask = get_region_mask(wavenumbers, reg)
        if mask.sum() < 3:
            continue
        seg = spectrum[mask]
        wn_seg = wavenumbers[mask]

        # Area (trapezoid)
        area = np.trapezoid(seg, wn_seg)
        features[r, 0] = area

        # Peak height and position
        peak_idx = np.argmax(seg)
        features[r, 1] = seg[peak_idx]
        features[r, 2] = wn_seg[peak_idx]

        # FWHM
        half_max = seg[peak_idx] / 2.0
        above = seg >= half_max
        if above.sum() >= 2:
            idx_above = np.where(above)[0]
            features[r, 3] = wn_seg[idx_above[-1]] - wn_seg[idx_above[0]]

        # Skewness
        m = seg.mean()
        s = seg.std()
        if s > 1e-10:
            features[r, 4] = np.mean(((seg - m) / s) ** 3)

    return features


def extract_derivative_features(spectrum, wavenumbers, n_seg=8):
    """
    Per-segment mean |d1|, mean |d2|.
    Returns shape: (n_seg, 2) flattened to (n_seg*2,)
    """
    seg_len = len(spectrum) // n_seg
    feats = np.zeros((n_seg, 2), dtype=np.float32)
    for i in range(n_seg):
        lo = i * seg_len
        hi = lo + seg_len if i < n_seg - 1 else len(spectrum)
        seg = spectrum[lo:hi]
        if len(seg) < 3:
            continue
        d1 = np.abs(np.diff(seg))
        d2 = np.abs(np.diff(seg, n=2))
        feats[i, 0] = d1.mean()
        feats[i, 1] = d2.mean() if len(d2) > 0 else 0
    return feats.flatten()


def extract_all_bond_features(X, wavenumbers, regions=None):
    """Extract bond features for a batch."""
    if regions is None:
        regions = REGIONS
    all_feats = []
    for i in range(len(X)):
        bf = extract_bond_features(X[i], wavenumbers, regions).flatten()
        df = extract_derivative_features(X[i], wavenumbers)
        all_feats.append(np.concatenate([bf, df]))
    return np.array(all_feats, dtype=np.float32)
