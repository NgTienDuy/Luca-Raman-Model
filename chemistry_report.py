"""
chemistry_report.py — Trích xuất, chú giải và so sánh thông tin Hóa-Lý
Mỗi model tạo profile riêng + commentary dựa trên dữ liệu thực của model đó
"""
import numpy as np, os, json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectral_knowledge import get_active_regions, get_region_indices, extract_bond_features

AA_REFERENCE = {
    "Alanine":      {"pI":6.00,"hydro":1.8, "mw":89.09, "aromatic":False},
    "Asparagine":   {"pI":5.41,"hydro":-3.5,"mw":132.12,"aromatic":False},
    "Aspartic Acid":{"pI":2.77,"hydro":-3.5,"mw":133.10,"aromatic":False},
    "Glutamic Acid":{"pI":3.22,"hydro":-3.5,"mw":147.13,"aromatic":False},
    "Histidine":    {"pI":7.59,"hydro":-3.2,"mw":155.15,"aromatic":True},
    "Glucosamine":  {"pI":6.73,"hydro":-3.0,"mw":215.63,"aromatic":False},
}

@dataclass
class ChemistryProfile:
    pH_score: float = 0.0; pH_label: str = "trung tính"
    polarity: float = 0.0; hydrophilicity: float = 0.0; hydrophobicity: float = 0.0
    bond_strength: float = 0.0; aromaticity: float = 0.0
    amide_I_III_ratio: float = 0.0; amide_I_pos_cm: float = 0.0
    secondary_structure: str = "unknown"
    aliphatic_index: float = 0.0; carboxylate_ratio: float = 0.0
    amine_ratio: float = 0.0; CN_CC_ratio: float = 0.0
    crystallinity_proxy: float = 0.0; spectral_entropy: float = 0.0
    dominant_bond: str = ""; top3_bonds: List[str] = field(default_factory=list)
    estimated_pI: float = 0.0; estimated_mw: float = 0.0
    estimated_hydrophobicity: float = 0.0
    bond_intensities: Dict[str,float] = field(default_factory=dict)
    bond_weights:     Dict[str,float] = field(default_factory=dict)
    # Commentary per property (set by model-specific logic)
    comments: Dict[str,str] = field(default_factory=dict)

def _comment_pH(v):
    if v > 0.3:  return f"Kiềm rõ rệt ({v:+.3f}) — NH₃⁺ chiếm ưu thế, pH ~7.5–9"
    if v > 0.1:  return f"Kiềm nhẹ ({v:+.3f}) — cân bằng amine/carboxyl, pH ~7–8"
    if v > -0.1: return f"Gần trung tính ({v:+.3f}) — cân bằng tốt, pH ~6.5–7.5"
    if v > -0.3: return f"Axit nhẹ ({v:+.3f}) — COO⁻ chiếm ưu thế, pH ~5–6.5"
    return f"Axit rõ ({v:+.3f}) — carboxylate cao, pH ~3–5"

def _comment_polarity(v):
    if v > 0.6: return f"Cao ({v:.3f}) — nhiều C=O, N–H, phân cực mạnh"
    if v > 0.45: return f"Trung bình ({v:.3f}) — có C=O, N–H nhưng cân bằng"
    return f"Thấp ({v:.3f}) — chủ yếu C–C, C–H không phân cực"

def _comment_hydro(v):
    if v > 0.55: return f"Cao ({v:.3f}) — ưa nước mạnh, tan tốt trong nước"
    if v > 0.40: return f"Trung bình ({v:.3f}) — tương đối ưa nước"
    return f"Thấp ({v:.3f}) — nhiều mạch aliphatic, kỵ nước"

def _comment_strength(v):
    if v > 0.7: return f"Cao ({v:.3f}) — liên kết C=O, C=C bền vững"
    if v > 0.5: return f"Trung bình ({v:.3f}) — C–C và C–N chủ đạo"
    return f"Thấp ({v:.3f}) — chủ yếu liên kết đơn yếu"

def _comment_arom(v):
    if v > 0.05: return f"Phát hiện rõ ({v:.4f}) — vòng imidazole His hoặc pyranose"
    if v > 0.01: return f"Phát hiện nhẹ ({v:.4f}) — có thể có vòng thơm"
    return f"Không phát hiện ({v:.4f}) — hỗn hợp chủ yếu aliphatic"

def _comment_amide(v):
    if v > 1.3: return f"Cao ({v:.3f}) — chuỗi α-helix hoặc C=O tự do chiếm ưu thế"
    if v > 0.8: return f"Cân bằng ({v:.3f}) — cấu trúc hỗn hợp"
    return f"Thấp ({v:.3f}) — β-sheet hoặc cấu trúc mở rộng"

def _comment_cryst(v):
    if v > 5: return f"Cao ({v:.2f}) — cấu trúc tinh thể rõ nét, đỉnh hẹp"
    if v > 2: return f"Trung bình ({v:.2f}) — tinh thể một phần"
    return f"Thấp ({v:.2f}) — cấu trúc vô định hình (amorphous)"

def extract_chemistry_profile(spectrum, wavenumbers, composition=None, label_names=None):
    regions = get_active_regions(wavenumbers)
    ridx    = get_region_indices(wavenumbers, regions)
    intensities={};fwhms={};peak_pos={}
    for br in regions:
        i0,i1=ridx[br.name]; seg=np.maximum(spectrum[i0:i1],0.); wn=wavenumbers[i0:i1]
        intensities[br.name]=float(seg.mean()) if len(seg)>0 else 0.
        ph=seg.max()
        if ph>0:
            above=np.where(seg>=ph/2.)[0]
            fwhms[br.name]=float(wn[above[-1]]-wn[above[0]]) if len(above)>=2 else 0.
            peak_pos[br.name]=float(wn[seg.argmax()])
        else: fwhms[br.name]=0.; peak_pos[br.name]=float((br.wn_min+br.wn_max)/2)
    total=sum(intensities.values())+1e-10; W={k:v/total for k,v in intensities.items()}
    polarity  =sum(W[br.name]*br.polarity       for br in regions)
    hydro     =sum(W[br.name]*br.hydrophilicity for br in regions)
    strength  =sum(W[br.name]*br.bond_strength  for br in regions)
    ph_raw    =sum(W[br.name]*br.ph_effect      for br in regions)
    ph_score  =float(np.tanh(ph_raw*3.0))
    ph_label  ="kiềm" if ph_score>0.15 else ("axit" if ph_score<-0.15 else "trung tính")
    arom_val  =sum(W.get(br.name,0.) for br in regions if br.bond_type=="aromatic")
    amide_I   =sum(intensities.get(br.name,0.) for br in regions if "amide_I"  in br.bond_type)+1e-10
    amide_III =sum(intensities.get(br.name,0.) for br in regions if "amide_III" in br.bond_type)+1e-10
    ratio     =float(amide_I/amide_III)
    amide_pos =next((peak_pos[br.name] for br in regions if "amide_I" in br.bond_type and br.name in peak_pos),0.)
    sec_struct=("α-helix dominant" if amide_pos>1665 else
                "β-sheet dominant" if amide_pos>1620 else
                "mixed"             if amide_pos>0    else "unknown")
    aliph_val =sum(W.get(br.name,0.) for br in regions if br.bond_type=="aliphatic")
    coo_val   =sum(W.get(br.name,0.) for br in regions if br.bond_type=="carboxylate")
    amine_val =sum(W.get(br.name,0.) for br in regions if "amine" in br.bond_type)
    cn_val    =sum(W.get(br.name,0.) for br in regions if br.bond_type=="amine")+1e-10
    cc_val    =sum(W.get(br.name,0.) for br in regions if br.bond_type=="aliphatic")+1e-10
    valid_fwhm=[v for v in fwhms.values() if v>5.]
    cryst=float(1./np.mean(valid_fwhm))*100 if valid_fwhm else 0.
    spec_pos=np.maximum(spectrum,0.); spec_pos/=spec_pos.sum()+1e-10
    spec_pos=spec_pos[spec_pos>0]
    entropy=float(-np.sum(spec_pos*np.log(spec_pos+1e-10)))
    sorted_bonds=sorted(W.items(),key=lambda x:x[1],reverse=True)
    dominant=sorted_bonds[0][0] if sorted_bonds else ""; top3=[n for n,_ in sorted_bonds[:3]]
    est_pI=est_mw=est_hydro=0.
    if composition is not None and label_names is not None:
        for j,name in enumerate(label_names):
            if name in AA_REFERENCE:
                w=float(composition[j])
                est_pI  +=w*AA_REFERENCE[name]["pI"]
                est_mw  +=w*AA_REFERENCE[name]["mw"]
                est_hydro+=w*AA_REFERENCE[name]["hydro"]
    comments={
        "pH":           _comment_pH(ph_score),
        "polarity":     _comment_polarity(polarity),
        "hydrophilicity":_comment_hydro(hydro),
        "bond_strength": _comment_strength(strength),
        "aromaticity":  _comment_arom(arom_val),
        "amide_ratio":  _comment_amide(ratio),
        "crystallinity":_comment_cryst(cryst),
    }
    return ChemistryProfile(
        pH_score=ph_score,pH_label=ph_label,polarity=polarity,
        hydrophilicity=hydro,hydrophobicity=float(1.-hydro),
        bond_strength=strength,aromaticity=arom_val,
        amide_I_III_ratio=ratio,amide_I_pos_cm=amide_pos,secondary_structure=sec_struct,
        aliphatic_index=aliph_val,carboxylate_ratio=coo_val,amine_ratio=amine_val,
        CN_CC_ratio=float(cn_val/cc_val),crystallinity_proxy=cryst,spectral_entropy=entropy,
        dominant_bond=dominant,top3_bonds=top3,
        estimated_pI=est_pI,estimated_mw=est_mw,estimated_hydrophobicity=est_hydro,
        bond_intensities=intensities,bond_weights=dict(W),comments=comments,
    )

def batch_chemistry(X, wavenumbers, Y=None, label_names=None):
    return [extract_chemistry_profile(s,wavenumbers,
                                      Y[i] if Y is not None else None,
                                      label_names) for i,s in enumerate(X)]

def mean_profile(profiles):
    if not profiles: return ChemistryProfile()
    keys=[k for k in ChemistryProfile.__dataclass_fields__ if isinstance(getattr(profiles[0],k),float)]
    avg=ChemistryProfile()
    for k in keys: setattr(avg,k,float(np.mean([getattr(p,k) for p in profiles])))
    avg.pH_label          =max(set(p.pH_label for p in profiles),key=lambda x:sum(1 for p in profiles if p.pH_label==x))
    avg.secondary_structure=max(set(p.secondary_structure for p in profiles),key=lambda x:sum(1 for p in profiles if p.secondary_structure==x))
    avg.dominant_bond     =max(set(p.dominant_bond for p in profiles),key=lambda x:sum(1 for p in profiles if p.dominant_bond==x))
    avg.top3_bonds        =profiles[0].top3_bonds
    all_keys=set()
    for p in profiles: all_keys.update(p.bond_intensities.keys())
    avg.bond_intensities={k:float(np.mean([p.bond_intensities.get(k,0.) for p in profiles])) for k in all_keys}
    avg.bond_weights    ={k:float(np.mean([p.bond_weights.get(k,0.)     for p in profiles])) for k in all_keys}
    # Regenerate comments from averaged values
    avg.comments={
        "pH":           _comment_pH(avg.pH_score),
        "polarity":     _comment_polarity(avg.polarity),
        "hydrophilicity":_comment_hydro(avg.hydrophilicity),
        "bond_strength": _comment_strength(avg.bond_strength),
        "aromaticity":  _comment_arom(avg.aromaticity),
        "amide_ratio":  _comment_amide(avg.amide_I_III_ratio),
        "crystallinity":_comment_cryst(avg.crystallinity_proxy),
    }
    return avg

def format_report(prof, model_name="", show_composition=True):
    """Tạo bảng báo cáo hóa-lý với cột Nhận xét đầy đủ"""
    lines=[f"{'═'*75}"]
    if model_name: lines.append(f"  Phân tích Hóa-Lý: {model_name}")
    lines.append(f"{'─'*75}")
    lines.append(f"  {'Tính chất':<22} {'Giá trị':>14}  Nhận xét")
    lines.append(f"  {'─'*70}")
    def row(label, val_str, comment=""):
        return f"  {label:<22} {val_str:>14}  {comment}"
    c=prof.comments
    lines.append(row("pH xu hướng",    f"{prof.pH_score:+.3f}",          c.get("pH","")))
    lines.append(row("Độ phân cực",    f"{prof.polarity:.3f}",            c.get("polarity","")))
    lines.append(row("Tính ưa nước",   f"{prof.hydrophilicity:.3f}",      c.get("hydrophilicity","")))
    lines.append(row("Bền liên kết",   f"{prof.bond_strength:.3f}",       c.get("bond_strength","")))
    lines.append(row("Tính thơm",      f"{prof.aromaticity:.4f}",         c.get("aromaticity","")))
    lines.append(f"  {'─'*70}")
    lines.append(row("Amide I/III",    f"{prof.amide_I_III_ratio:.3f}",   c.get("amide_ratio","")))
    if prof.amide_I_pos_cm>0:
        lines.append(row("Amide I pos",f"{prof.amide_I_pos_cm:.1f} cm⁻¹",""))
    lines.append(row("Cấu trúc bậc 2",f"{prof.secondary_structure[:18]:<18}",""))
    lines.append(f"  {'─'*70}")
    lines.append(row("Aliphatic idx",  f"{prof.aliphatic_index:.4f}",     "CH₂/CH₃, kỵ nước"))
    lines.append(row("Carboxylate",    f"{prof.carboxylate_ratio:.4f}",   "COO⁻ abundance"))
    lines.append(row("Amine ratio",    f"{prof.amine_ratio:.4f}",         "NH₃⁺/NH₂"))
    lines.append(row("Crystallinity",  f"{prof.crystallinity_proxy:.2f}", c.get("crystallinity","")))
    lines.append(row("Spectral ent.",  f"{prof.spectral_entropy:.3f}",    "cao=phổ phức tạp"))
    lines.append(f"  {'─'*70}")
    lines.append(f"  Liên kết chủ đạo : {prof.dominant_bond}")
    lines.append(f"  Top-3 bonds      : {', '.join(prof.top3_bonds)}")
    if show_composition and prof.estimated_pI>0:
        lines.append(f"  {'─'*70}")
        lines.append(row("Est. pI", f"{prof.estimated_pI:.2f}", "điểm đẳng điện từ tỉ lệ AA"))
        lines.append(row("Est. MW", f"{prof.estimated_mw:.1f} Da","từ tỉ lệ AA"))
        lines.append(row("Est. hydro",f"{prof.estimated_hydrophobicity:.2f}","Kyte-Doolittle"))
    lines.append(f"{'═'*75}")
    return "\n".join(lines)

def compare_models(model_profiles: Dict[str,ChemistryProfile], save_path=None):
    """So sánh cross-model với nhận xét giống/khác"""
    props=[
        ("pH score","pH_score","{:+.3f}","comments","pH"),
        ("Polarity","polarity","{:.3f}","comments","polarity"),
        ("Hydrophilicity","hydrophilicity","{:.3f}","comments","hydrophilicity"),
        ("Bond strength","bond_strength","{:.3f}","comments","bond_strength"),
        ("Aromaticity","aromaticity","{:.4f}","comments","aromaticity"),
        ("Amide I/III","amide_I_III_ratio","{:.3f}","comments","amide_ratio"),
        ("Crystallinity","crystallinity_proxy","{:.2f}","comments","crystallinity"),
        ("Entropy","spectral_entropy","{:.3f}","",""),
    ]
    names=list(model_profiles.keys()); col_w=max(len(n) for n in names)+2
    lines=[f"\n{'═'*90}","  SO SÁNH THÔNG TIN HÓA-LÝ GIỮA CÁC MÔ HÌNH",f"{'─'*90}"]
    header=f"  {'Tính chất':<22}"+"".join(f"{n:>{col_w}}" for n in names)
    lines.append(header); lines.append(f"  {'─'*85}")
    for label,key,fmt,ckey,cfield in props:
        row=f"  {label:<22}"
        for n in names:
            v=getattr(model_profiles[n],key,0.)
            row+=f"{fmt.format(float(v)):>{col_w}}"
        lines.append(row)
    lines.append(f"{'─'*90}")
    lines.append("  NHẬN XÉT SO SÁNH (tính chất chính):")
    for label,key,fmt,ckey,cfield in props:
        if not cfield: continue
        vals={}
        for n in names:
            prof=model_profiles[n]
            if ckey=="comments": vals[n]=prof.comments.get(cfield,"")
            else: vals[n]=getattr(prof,cfield,"")
        unique_vals=list(dict.fromkeys(vals.values()))
        lines.append(f"\n  [{label}]")
        for mname,comment in vals.items():
            lines.append(f"    {mname}: {comment}")
        if len(unique_vals)==1:
            lines.append(f"    → Tất cả mô hình ĐỒNG NHẤT: {unique_vals[0]}")
        else:
            # Find which models agree
            from collections import Counter
            cnt=Counter(vals.values()); most_common=cnt.most_common(1)[0]
            if most_common[1]>1:
                agree=[n for n,v in vals.items() if v==most_common[0]]
                lines.append(f"    → Đa số đồng thuận: {', '.join(agree)}")
            lines.append(f"    → Có {len(unique_vals)} quan điểm khác nhau")
    lines.append(f"\n{'═'*90}")
    text="\n".join(lines)
    if save_path:
        _plot_radar(model_profiles, save_path)
        _plot_bond_heatmap(model_profiles, save_path)
    return text

def _plot_radar(model_profiles, save_dir):
    props_radar=[
        ("pH\n(norm)","pH_score",lambda x:(x+1)/2),
        ("Polarity","polarity",lambda x:x),
        ("Hydrophilic","hydrophilicity",lambda x:x),
        ("Bond\nStrength","bond_strength",lambda x:x),
        ("Aromatic","aromaticity",lambda x:min(x*10,1)),
        ("Aliphatic","aliphatic_index",lambda x:min(x*5,1)),
        ("Carboxyl","carboxylate_ratio",lambda x:min(x*10,1)),
        ("Amine","amine_ratio",lambda x:min(x*10,1)),
        ("Crystallinity","crystallinity_proxy",lambda x:min(x/50,1)),
    ]
    labels=[p[0] for p in props_radar]; N=len(labels)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    colors=plt.cm.Set1(np.linspace(0,0.8,len(model_profiles)))
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    for (mname,prof),color in zip(model_profiles.items(),colors):
        vals=[p[2](getattr(prof,p[1],0.)) for p in props_radar]; vals+=vals[:1]
        ax.plot(angles,vals,"o-",lw=2,label=mname,color=color)
        ax.fill(angles,vals,alpha=0.08,color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,fontsize=9)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"],fontsize=7); ax.grid(True,alpha=0.3)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=9)
    ax.set_title("So sánh Tính chất Hóa-Lý giữa các Model",fontsize=12,pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"chemistry_comparison_radar.png"),dpi=120,bbox_inches="tight")
    plt.close()

def _plot_bond_heatmap(model_profiles, save_dir):
    all_bonds=[]
    for p in model_profiles.values(): all_bonds.extend(p.bond_weights.keys())
    bond_names=list(dict.fromkeys(all_bonds)); model_names=list(model_profiles.keys())
    data=np.zeros((len(bond_names),len(model_names)))
    for j,(mname,prof) in enumerate(model_profiles.items()):
        for i,bn in enumerate(bond_names): data[i,j]=prof.bond_weights.get(bn,0.)
    fig,ax=plt.subplots(figsize=(max(6,len(model_names)*1.5),max(5,len(bond_names)*0.5)))
    im=ax.imshow(data,aspect="auto",cmap="YlOrRd",vmin=0)
    ax.set_xticks(range(len(model_names))); ax.set_xticklabels(model_names,rotation=30,ha="right",fontsize=9)
    ax.set_yticks(range(len(bond_names)));  ax.set_yticklabels(bond_names,fontsize=8)
    plt.colorbar(im,ax=ax,label="Bond weight")
    ax.set_title("Bond Region Weights × Model",fontsize=12)
    for i in range(len(bond_names)):
        for j in range(len(model_names)):
            ax.text(j,i,f"{data[i,j]:.3f}",ha="center",va="center",fontsize=7,
                    color="white" if data[i,j]>0.06 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"chemistry_bond_heatmap.png"),dpi=120,bbox_inches="tight")
    plt.close()

def save_chemistry_json(profiles_dict, save_path):
    out={}
    for name,prof in profiles_dict.items():
        d=asdict(prof); out[name]=d
    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2,ensure_ascii=False,
                  default=lambda x:float(x) if hasattr(x,"__float__") else str(x))

def plot_bond_contribution(prof, model_name, save_path):
    from spectral_knowledge import get_active_regions
    from matplotlib.patches import Patch
    bonds=sorted(prof.bond_weights.items(),key=lambda x:x[1],reverse=True)
    names=[b[0] for b in bonds]; weights=[b[1] for b in bonds]
    type_colors={"amine":"#E8593C","carboxylate":"#3B8BD4","amide_I":"#534AB7",
                 "amide_III":"#9F77DD","aliphatic":"#888780","aromatic":"#EF9F27",
                 "amino_frame":"#1D9E75","polar":"#D4537E"}
    dummy_wns=np.linspace(267,2004,1024)
    regions=get_active_regions(dummy_wns); type_map={br.name:br.bond_type for br in regions}
    colors=[type_colors.get(type_map.get(n,""),"#B4B2A9") for n in names]
    fig,ax=plt.subplots(figsize=(10,5))
    bars=ax.bar(names,weights,color=colors,edgecolor="white",linewidth=0.5)
    ax.set_xlabel("Bond Region"); ax.set_ylabel("Weight")
    ax.set_title(f"{model_name} — Bond Region Contributions")
    ax.tick_params(axis="x",rotation=45); ax.grid(True,alpha=0.3,axis="y")
    for bar,w in zip(bars,weights):
        ax.text(bar.get_x()+bar.get_width()/2,w+0.001,f"{w:.3f}",ha="center",va="bottom",fontsize=8)
    seen={}
    for n,c in zip(names,colors):
        bt=type_map.get(n,"other")
        if bt not in seen: seen[bt]=c
    ax.legend(handles=[Patch(color=c,label=bt) for bt,c in seen.items()],fontsize=8,loc="upper right")
    plt.tight_layout(); plt.savefig(save_path,dpi=100,bbox_inches="tight"); plt.close()
