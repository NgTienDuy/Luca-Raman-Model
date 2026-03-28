"""
spectral_knowledge.py — Tri thức hóa học phổ Raman axit amin
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dải dữ liệu THỰC: 801.62 – 931.28 cm⁻¹ (~130 cm⁻¹)

Các đỉnh Raman quan trọng trong dải 800–931 cm⁻¹ cho amino acids:
  ~830    C-C-N bending / Tyr ring breathing
  ~850    C-C stretch, proline ring
  ~878    Trp C3-C3a stretch (indole)
  ~900    C-C stretch aliphatic
  ~920    C-N stretch α-amino
  ~930    C-C-N asymmetric

Bond regions được định nghĩa ĐỘNG dựa trên wavenumber thực tế
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class BondRegion:
    name: str
    wn_min: float
    wn_max: float
    bond_type: str
    polarity: float        # 0–1
    bond_strength: float   # 0–1
    hydrophilicity: float  # 0–1
    ph_effect: float       # -1 (acid) to +1 (base)
    description: str = ""


# ─── Bond regions tĩnh (400–1800 cm⁻¹) — dùng khi có dải rộng ────────────────
BOND_REGIONS_FULL: List[BondRegion] = [
    BondRegion("C-C-C",      400,  500,  "skeletal",    0.10, 0.60, 0.10,  0.00, "Khung carbon"),
    BondRegion("C-S/C-Cl",   500,  600,  "C-hetero",    0.30, 0.55, 0.20,  0.00, "C với dị nguyên tố"),
    BondRegion("C-C-N",      600,  750,  "amino_frame", 0.45, 0.65, 0.40,  0.05, "Khung amino acid"),
    BondRegion("C-C",        750,  900,  "aliphatic",   0.10, 0.60, 0.05,  0.00, "Mạch carbon, kỵ nước"),
    BondRegion("C-N",        900, 1000,  "amine",       0.50, 0.70, 0.55,  0.10, "Nhóm amine alpha"),
    BondRegion("C-O/C-N+",  1000, 1120,  "polar",       0.75, 0.72, 0.70, -0.20, "Đường, rượu; pH thấp"),
    BondRegion("C-H bend",  1120, 1250,  "aliphatic",   0.15, 0.55, 0.10,  0.00, "Biến dạng C-H"),
    BondRegion("N-H/C-H",   1250, 1380,  "amide_III",   0.55, 0.68, 0.60,  0.05, "Amide III"),
    BondRegion("COO-",       1380, 1430,  "carboxylate", 0.90, 0.75, 0.85,  0.50, "Carboxylate"),
    BondRegion("CH2/CH3",    1430, 1480,  "aliphatic",   0.08, 0.52, 0.05,  0.00, "Mạch aliphatic"),
    BondRegion("NH3+/NH2",   1480, 1560,  "amine_bend",  0.80, 0.70, 0.75, -0.40, "Amine protonated"),
    BondRegion("C=C/C=N",    1560, 1640,  "aromatic",    0.60, 0.85, 0.50,  0.15, "Vòng thơm"),
    BondRegion("C=O",        1640, 1720,  "amide_I",     0.95, 0.80, 0.80,  0.00, "Amide I"),
]

# ─── Bond regions cho dải 801–931 cm⁻¹ ────────────────────────────────────────
BOND_REGIONS_NARROW: List[BondRegion] = [
    BondRegion("CC-ring",    801,  825,  "ring",        0.30, 0.65, 0.30, 0.00, "C-C ring breathing, Pro"),
    BondRegion("CCN-bend",   825,  845,  "amino_frame", 0.45, 0.65, 0.40, 0.05, "C-C-N bending, Tyr"),
    BondRegion("CC-aliphat", 845,  875,  "aliphatic",   0.10, 0.60, 0.05, 0.00, "C-C aliphatic stretch"),
    BondRegion("Trp-indole", 875,  895,  "aromatic",    0.60, 0.80, 0.45, 0.10, "Trp C3-C3a, indole ring"),
    BondRegion("CC-stretch", 895,  910,  "aliphatic",   0.12, 0.62, 0.08, 0.00, "C-C stretch backbone"),
    BondRegion("CN-alpha",   910,  931,  "amine",       0.50, 0.70, 0.55, 0.10, "C-N α-amino stretch"),
]


def get_active_regions(wavenumbers: np.ndarray) -> List[BondRegion]:
    """
    Tự động chọn bond regions phù hợp với dải wavenumber thực tế.
    Ưu tiên narrow regions nếu dải dữ liệu < 200 cm⁻¹.
    """
    wn_min, wn_max = float(wavenumbers.min()), float(wavenumbers.max())
    wn_range = wn_max - wn_min

    if wn_range < 200:
        # Dải hẹp: dùng narrow regions, lọc chỉ các region có overlap
        active = [br for br in BOND_REGIONS_NARROW
                  if br.wn_min < wn_max and br.wn_max > wn_min]
        if not active:
            active = _make_dynamic_regions(wn_min, wn_max, n=6)
    else:
        active = [br for br in BOND_REGIONS_FULL
                  if br.wn_min < wn_max and br.wn_max > wn_min]
        if not active:
            active = _make_dynamic_regions(wn_min, wn_max, n=8)

    return active


def _make_dynamic_regions(wn_min: float, wn_max: float,
                           n: int = 8) -> List[BondRegion]:
    """Tạo n vùng đều nhau khi không có prior hóa học."""
    edges = np.linspace(wn_min, wn_max, n+1)
    regions = []
    for i in range(n):
        regions.append(BondRegion(
            name=f"seg{i+1}",
            wn_min=float(edges[i]),
            wn_max=float(edges[i+1]),
            bond_type="unknown",
            polarity=0.5, bond_strength=0.6,
            hydrophilicity=0.4, ph_effect=0.0,
            description=f"Auto segment {i+1}",
        ))
    return regions


def get_region_indices(wavenumbers: np.ndarray,
                       regions: Optional[List[BondRegion]] = None
                       ) -> Dict[str, Tuple[int, int]]:
    """Trả về (start_idx, end_idx) cho mỗi bond region."""
    if regions is None:
        regions = get_active_regions(wavenumbers)
    out = {}
    for br in regions:
        i0 = int(np.searchsorted(wavenumbers, br.wn_min))
        i1 = int(np.searchsorted(wavenumbers, br.wn_max))
        i0 = max(0, min(i0, len(wavenumbers)-1))
        i1 = max(i0+1, min(i1, len(wavenumbers)))
        out[br.name] = (i0, i1)
    return out


# ─── Feature extraction ────────────────────────────────────────────────────────
def extract_bond_features(spectrum: np.ndarray,
                           wavenumbers: np.ndarray,
                           regions: Optional[List[BondRegion]] = None
                           ) -> np.ndarray:
    """
    Trích xuất 5 đặc trưng × n_regions = 5n features.
    Tự động dùng regions phù hợp với dải wavenumber.
    """
    if regions is None:
        regions = get_active_regions(wavenumbers)
    ridx = get_region_indices(wavenumbers, regions)
    feats = []
    for br in regions:
        i0, i1 = ridx[br.name]
        seg = np.maximum(spectrum[i0:i1], 0.0)
        wn_seg = wavenumbers[i0:i1]
        total = seg.sum() + 1e-10

        area  = float(np.trapezoid(seg, wn_seg)) if (len(wn_seg)>1 and hasattr(np,'trapezoid')) else (float(np.trapz(seg, wn_seg)) if (len(wn_seg)>1 and hasattr(np,'trapz')) else float(total))
        ph    = float(seg.max())
        pp    = float(wn_seg[np.argmax(seg)]) if len(seg)>0 else float((br.wn_min+br.wn_max)/2)

        above = np.where(seg >= ph/2)[0]
        fwhm  = float(wn_seg[above[-1]]-wn_seg[above[0]]) if len(above)>=2 else 0.

        mean_wn = float(np.sum(wn_seg*seg)/total)
        std_wn  = float(np.sqrt(np.sum((wn_seg-mean_wn)**2*seg/total)))+1e-10
        skew    = float(np.sum(((wn_seg-mean_wn)/std_wn)**3*seg/total))

        feats.extend([area, ph, pp, fwhm, skew])
    return np.array(feats, dtype=np.float32)


def extract_derivative_features(spectrum: np.ndarray,
                                  wavenumbers: np.ndarray,
                                  n_seg: int = 8) -> np.ndarray:
    """16 features: mean|∇¹| và mean|∇²| trong n_seg vùng."""
    d1 = np.abs(np.gradient(spectrum))
    d2 = np.abs(np.gradient(d1))
    segs = np.array_split(np.arange(len(spectrum)), n_seg)
    feats = []
    for idx in segs:
        feats.append(float(d1[idx].mean()))
        feats.append(float(d2[idx].mean()))
    return np.array(feats, dtype=np.float32)


def extract_region_stats(spectrum: np.ndarray,
                          wavenumbers: np.ndarray,
                          regions: Optional[List[BondRegion]] = None
                          ) -> np.ndarray:
    """5 thống kê × n_regions features."""
    if regions is None:
        regions = get_active_regions(wavenumbers)
    ridx = get_region_indices(wavenumbers, regions)
    feats = []
    for br in regions:
        i0, i1 = ridx[br.name]
        seg = spectrum[i0:i1] if i1>i0 else np.array([0.])
        feats.extend([
            float(seg.mean()), float(seg.std()),
            float(seg.max()),  float(seg.min()),
            float((seg**2).sum()),
        ])
    return np.array(feats, dtype=np.float32)


def compute_physicochemical_profile(spectrum: np.ndarray,
                                     wavenumbers: np.ndarray,
                                     regions: Optional[List[BondRegion]] = None
                                     ) -> Dict:
    """Tính tính chất hóa lý từ phổ."""
    if regions is None:
        regions = get_active_regions(wavenumbers)
    ridx = get_region_indices(wavenumbers, regions)
    intensities = {}
    for br in regions:
        i0,i1 = ridx[br.name]
        intensities[br.name] = float(np.maximum(spectrum[i0:i1],0).mean()) if i1>i0 else 0.

    total = sum(intensities.values())+1e-10
    W = {k:v/total for k,v in intensities.items()}

    polarity  = sum(W[br.name]*br.polarity       for br in regions)
    hydro     = sum(W[br.name]*br.hydrophilicity  for br in regions)
    ph_score  = float(np.tanh(sum(W[br.name]*br.ph_effect  for br in regions)))
    strength  = sum(W[br.name]*br.bond_strength   for br in regions)

    return {
        "polarity":         polarity,
        "hydrophilicity":   hydro,
        "ph_score":         ph_score,
        "bond_strength":    strength,
        "bond_intensities": intensities,
        "bond_weights":     W,
    }


def interpret_profile(profile: Dict) -> str:
    lines = [
        "─── Phân tích Hóa Lý Phổ ────────────────",
        f"  pH xu hướng : {'kiềm' if profile['ph_score']>0.1 else 'axit' if profile['ph_score']<-0.1 else 'trung tính'} ({profile['ph_score']:+.3f})",
        f"  Độ phân cực : {profile['polarity']:.3f}",
        f"  Tính ưa nước: {profile['hydrophilicity']:.3f}",
        f"  Bền liên kết: {profile['bond_strength']:.3f}",
        "  Top-3 bond region:",
    ]
    top3 = sorted(profile["bond_weights"].items(), key=lambda x:x[1], reverse=True)[:3]
    for name,w in top3:
        lines.append(f"    {name:<15}: {w:.4f}")
    return "\n".join(lines)


# ─── Amino acid profiles ───────────────────────────────────────────────────────
@dataclass
class AminoAcidProfile:
    name: str
    formula: str
    mw: float
    pI: float
    peaks_800_931: List[float]   # peaks trong dải 800-931
    peak_rel_int:  List[float]
    hydrophobicity: float

AA_PROFILES = {
    "Alanine": AminoAcidProfile(
        "DL-Alanine","C3H7NO2",89.09,6.00,
        [852, 895, 920],[0.7,0.8,0.6],1.8),
    "Asparagine": AminoAcidProfile(
        "L-Asparagine","C4H8N2O3",132.12,5.41,
        [830, 875, 915],[0.6,0.7,0.5],-3.5),
    "Aspartic Acid": AminoAcidProfile(
        "L-Aspartic acid","C4H7NO4",133.10,2.77,
        [840, 878, 918],[0.8,0.6,0.7],-3.5),
    "Glutamic Acid": AminoAcidProfile(
        "L-Glutamic acid","C5H9NO4",147.13,3.22,
        [835, 870, 912],[0.7,0.8,0.6],-3.5),
    "Histidine": AminoAcidProfile(
        "L-Histidine","C6H9N3O2",155.15,7.59,
        [822, 855, 927],[0.9,0.7,0.8],-3.2),
    "Glucosamine": AminoAcidProfile(
        "D-Glucosamine-HCl","C6H14ClNO5",215.63,6.73,
        [844, 889, 916],[0.8,0.9,0.7],-3.0),
}
