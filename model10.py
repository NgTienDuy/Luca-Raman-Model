"""
Model 10: Radial Exhaustive Information Explorer (RIER).
8 spoke modules (parallel branches) with gated fusion.
Scientific discovery engine — extracts maximum information.
"""
import os, time, logging, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA, NMF

import config as C
from utils import (preprocess_batch, compute_metrics, softmax_normalize, plot_scatter_aggregated, plot_scatter_raw,
                   plot_loss_curve, save_results, set_seeds)
from spectral_knowledge import REGIONS, N_REGIONS, extract_bond_features, extract_derivative_features
from model_helpers import post_process_advanced_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
MODEL_NUM = 10


# ═══════════════════════════════════════════════════════════════════════════
#  SPOKE MODULES
# ═══════════════════════════════════════════════════════════════════════════

class VAESpoke(nn.Module):
    """Variational Autoencoder spoke."""
    def __init__(self, in_dim, z_dim=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, z_dim)
        self.logvar = nn.Linear(64, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, in_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.dec(z)
        return z, mu, logvar, recon


class LinearSpoke(nn.Module):
    """Simple linear projection spoke (PCA-like)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class MomentsSpoke(nn.Module):
    """Statistical moments per segment."""
    def __init__(self, in_dim, n_seg=8):
        super().__init__()
        self.n_seg = n_seg
        self.out_dim = n_seg * 4  # mean, std, skew, kurt per segment
        self.proj = nn.Linear(self.out_dim, 32)

    def forward(self, x):
        B, D = x.shape
        seg_len = D // self.n_seg
        moments = []
        for i in range(self.n_seg):
            lo = i * seg_len
            hi = lo + seg_len if i < self.n_seg - 1 else D
            seg = x[:, lo:hi]
            m = seg.mean(dim=1, keepdim=True)
            s = seg.std(dim=1, keepdim=True) + 1e-8
            skew = ((seg - m) ** 3).mean(dim=1, keepdim=True) / (s ** 3)
            kurt = ((seg - m) ** 4).mean(dim=1, keepdim=True) / (s ** 4) - 3
            moments.extend([m, s, skew, kurt])
        moments = torch.cat(moments, dim=1)
        return self.proj(moments)


class XCorrSpoke(nn.Module):
    """Cross-correlation between first and second half."""
    def __init__(self, in_dim, out_dim=16):
        super().__init__()
        half = in_dim // 2
        self.proj = nn.Linear(half, out_dim)
        self.half = half

    def forward(self, x):
        h1 = x[:, :self.half]
        h2 = x[:, self.half:self.half*2]
        # Element-wise correlation
        corr = h1 * h2
        return self.proj(corr)


# ═══════════════════════════════════════════════════════════════════════════
#  RIER MODEL
# ═══════════════════════════════════════════════════════════════════════════

class RIER(nn.Module):
    def __init__(self, in_dim, bond_feat_dim, fft_dim=32, pca_dim=30,
                 z_dim=32, nmf_dim=8):
        super().__init__()
        self.in_dim = in_dim

        # 8 spokes
        self.vae = VAESpoke(in_dim, z_dim)           # Spoke 1: VAE
        self.pca_spoke = LinearSpoke(in_dim, pca_dim) # Spoke 2: PCA-like
        self.fft_spoke = LinearSpoke(fft_dim, 32)     # Spoke 3: FFT
        self.bond_spoke = LinearSpoke(bond_feat_dim, 32)  # Spoke 4: Bond stats
        self.deriv_spoke = LinearSpoke(16, 16)        # Spoke 5: Derivatives
        self.nmf_spoke = LinearSpoke(nmf_dim, 16)     # Spoke 6: NMF
        self.moments_spoke = MomentsSpoke(in_dim)     # Spoke 7: Moments
        self.xcorr_spoke = XCorrSpoke(in_dim, 16)    # Spoke 8: XCorr

        # Spoke output dimensions
        spoke_dims = [z_dim, pca_dim, 32, 32, 16, 16, 32, 16]
        total_dim = sum(spoke_dims)

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

        # Spoke projections to common dim
        self.spoke_projs = nn.ModuleList([
            nn.Linear(d, 32) for d in spoke_dims
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(32 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, C.NUM_AA)

    def forward(self, x, fft_feat, bond_feat, deriv_feat, nmf_feat):
        # Compute all spokes
        z_vae, mu, logvar, recon = self.vae(x)
        z_pca = self.pca_spoke(x)
        z_fft = self.fft_spoke(fft_feat)
        z_bond = self.bond_spoke(bond_feat)
        z_deriv = self.deriv_spoke(deriv_feat)
        z_nmf = self.nmf_spoke(nmf_feat)
        z_mom = self.moments_spoke(x)
        z_xcorr = self.xcorr_spoke(x)

        spokes = [z_vae, z_pca, z_fft, z_bond, z_deriv, z_nmf, z_mom, z_xcorr]

        # Concatenate for gate
        z_cat = torch.cat(spokes, dim=-1)
        gates = self.gate(z_cat)  # (B, 8)

        # Project and gate
        projected = []
        for i, (z, proj) in enumerate(zip(spokes, self.spoke_projs)):
            projected.append(gates[:, i:i+1] * proj(z))

        fused = torch.cat(projected, dim=-1)
        feat = self.fusion(fused)
        pred = F.softmax(self.head(feat), dim=-1)

        return pred, mu, logvar, recon, gates


def _compute_fft_features(X, n_freq=32):
    """FFT magnitude features (low frequencies)."""
    fft_feats = np.zeros((len(X), n_freq), dtype=np.float32)
    for i in range(len(X)):
        fft = np.fft.rfft(X[i])
        mag = np.abs(fft)[:n_freq]
        if len(mag) < n_freq:
            mag = np.pad(mag, (0, n_freq - len(mag)))
        fft_feats[i] = mag
    return fft_feats


def _compute_nmf_features(X_train, X, n_comp=8):
    """NMF features."""
    X_nn = np.maximum(X_train, 0) + 1e-10
    nmf = NMF(n_components=n_comp, max_iter=200, random_state=C.SEED, init='nndsvda')
    nmf.fit(X_nn)
    X_nn2 = np.maximum(X, 0) + 1e-10
    return nmf.transform(X_nn2).astype(np.float32), nmf


def _compute_deriv_features(X, wavenumbers):
    """Derivative features."""
    feats = np.zeros((len(X), 16), dtype=np.float32)
    for i in range(len(X)):
        feats[i] = extract_derivative_features(X[i], wavenumbers)
    return feats


def _compute_bond_features(X, wavenumbers):
    """Bond statistics."""
    feats = []
    for i in range(len(X)):
        bf = extract_bond_features(X[i], wavenumbers, REGIONS).flatten()
        feats.append(bf)
    return np.array(feats, dtype=np.float32)


def plot_radial_diagram(gate_weights, save_dir, model_num):
    """Plot flower-shaped DAG: INPUT → 8 spokes → FUSION, line width ∝ gate weight."""
    spoke_names = ['VAE', 'PCA', 'FFT', 'BondStats', 'Derivatives', 'NMF', 'Moments', 'XCorr']
    spoke_methods = ['Autoencoder', 'Linear proj', 'Fourier mag', 'Bond features',
                     'd1,d2 segments', 'Decomposition', 'Statistics', 'Cross-corr']
    spoke_dims = [32, 30, 32, 50, 16, 8, 32, 16]
    spoke_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                    '#1abc9c', '#e67e22', '#3498db', '#e91e63']
    mean_gates = gate_weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central INPUT node
    cx, cy = 0, 0
    circ = plt.Circle((cx, cy), 0.22, color='#3498db', zorder=10)
    ax.add_patch(circ)
    ax.text(cx, cy + 0.04, 'INPUT', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=11)
    ax.text(cx, cy - 0.06, 'Spectrum', ha='center', va='center', fontsize=8,
            color='white', zorder=11)

    # FUSION node at bottom
    fx, fy = 0, -1.35
    circ_f = plt.Circle((fx, fy), 0.18, color='#2c3e50', zorder=10)
    ax.add_patch(circ_f)
    ax.text(fx, fy + 0.03, 'FUSION', ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=11)
    ax.text(fx, fy - 0.06, '→ŷ', ha='center', va='center', fontsize=9,
            color='#ecf0f1', zorder=11)

    n = len(spoke_names)
    angles = np.linspace(np.pi * 0.15, np.pi * 0.85, n)  # upper semicircle

    for i in range(n):
        ang = angles[i]
        r_spoke = 1.1
        sx = r_spoke * np.cos(ang)
        sy = r_spoke * np.sin(ang)

        g = mean_gates[i]
        lw = max(1.0, g * 8)  # line width ∝ gate weight
        alpha = max(0.25, g)

        # Line: center → spoke
        ax.plot([cx, sx], [cy, sy], '-', color=spoke_colors[i],
                linewidth=lw, alpha=alpha, zorder=3)
        # Line: spoke → fusion
        ax.plot([sx, fx], [sy, fy], '-', color=spoke_colors[i],
                linewidth=lw * 0.6, alpha=alpha * 0.7, zorder=3)

        # Spoke node
        node_r = 0.12 + g * 0.06
        circ_s = plt.Circle((sx, sy), node_r, color=spoke_colors[i],
                            zorder=10, alpha=0.9)
        ax.add_patch(circ_s)
        ax.text(sx, sy + 0.02, spoke_names[i], ha='center', va='center',
                fontsize=8, fontweight='bold', color='white', zorder=11)
        ax.text(sx, sy - 0.06, f'd={spoke_dims[i]}', ha='center', va='center',
                fontsize=6, color='white', zorder=11)

        # Gate weight label
        lx = sx + 0.25 * np.cos(ang)
        ly = sy + 0.25 * np.sin(ang)
        ax.text(lx, ly, f'g={g:.2f}', fontsize=7, ha='center', va='center',
                color=spoke_colors[i], fontweight='bold')

    ax.set_title(f"Model {model_num}: RIER — Radial Exhaustive Explorer\n"
                 f"Line width ∝ gate weight", fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    out = os.path.join(save_dir, f"model{model_num:02d}_radial.png")
    fig.savefig(out, dpi=C.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved {out}")


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINTS_DIR, f"model{MODEL_NUM:02d}.pt")
    sid_test = kw.get("sid_test")

    log.info("M10: Preprocessing and feature extraction...")
    Xtr = preprocess_batch(X_train, 'sg_snv')
    Xv = preprocess_batch(X_val, 'sg_snv')
    Xt = preprocess_batch(X_test, 'sg_snv')
    D = Xtr.shape[1]

    # FFT features
    fft_tr = _compute_fft_features(Xtr)
    fft_v = _compute_fft_features(Xv)
    fft_t = _compute_fft_features(Xt)

    # Bond features
    bond_tr = _compute_bond_features(Xtr, wavenumbers)
    bond_v = _compute_bond_features(Xv, wavenumbers)
    bond_t = _compute_bond_features(Xt, wavenumbers)

    # Derivative features
    deriv_tr = _compute_deriv_features(Xtr, wavenumbers)
    deriv_v = _compute_deriv_features(Xv, wavenumbers)
    deriv_t = _compute_deriv_features(Xt, wavenumbers)

    # NMF features
    log.info("M10: Computing NMF features...")
    nmf_tr, nmf_model = _compute_nmf_features(Xtr, Xtr, C.RIER_NMF_K)
    nmf_v = nmf_model.transform(np.maximum(Xv, 0) + 1e-10).astype(np.float32)
    nmf_t = nmf_model.transform(np.maximum(Xt, 0) + 1e-10).astype(np.float32)

    bond_dim = bond_tr.shape[1]

    # Tensors
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)

    fft_tr_t = torch.tensor(fft_tr, dtype=torch.float32)
    fft_v_t = torch.tensor(fft_v, dtype=torch.float32)
    fft_t_t = torch.tensor(fft_t, dtype=torch.float32)
    bond_tr_t = torch.tensor(bond_tr, dtype=torch.float32)
    bond_v_t = torch.tensor(bond_v, dtype=torch.float32)
    bond_t_t = torch.tensor(bond_t, dtype=torch.float32)
    deriv_tr_t = torch.tensor(deriv_tr, dtype=torch.float32)
    deriv_v_t = torch.tensor(deriv_v, dtype=torch.float32)
    deriv_t_t = torch.tensor(deriv_t, dtype=torch.float32)
    nmf_tr_t = torch.tensor(nmf_tr, dtype=torch.float32)
    nmf_v_t = torch.tensor(nmf_v, dtype=torch.float32)
    nmf_t_t = torch.tensor(nmf_t, dtype=torch.float32)

    model = RIER(D, bond_dim, fft_dim=32, pca_dim=30,
                 z_dim=C.RIER_Z_DIM, nmf_dim=C.RIER_NMF_K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    ds = TensorDataset(Xtr_t, fft_tr_t, bond_tr_t, deriv_tr_t, nmf_tr_t, Ytr_t)
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True)

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_ctr = 0

    log.info("M10: Training RIER...")
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        ep_loss = 0
        nb = 0

        for xb, fb, bb, db, nb_f, yb in loader:
            pred, mu, logvar, recon, gates = model(xb, fb, bb, db, nb_f)

            # KL div for composition
            l_kl = F.kl_div(torch.clamp(pred, 1e-8).log(),
                            torch.clamp(yb, 1e-8), reduction='batchmean')

            # VAE reconstruction
            l_recon = F.mse_loss(recon, xb)

            # VAE KL
            l_vae_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Spoke diversity (encourage different spokes to activate)
            l_div = -torch.mean(gates.std(dim=0))

            loss = l_kl + 0.01 * l_recon + 0.001 * l_vae_kl + 0.01 * l_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vp, _, _, _, _ = model(Xv_t, fft_v_t, bond_v_t, deriv_v_t, nmf_v_t)
            vl = F.kl_div(torch.clamp(vp, 1e-8).log(),
                          torch.clamp(Yv_t, 1e-8), reduction='batchmean').item()
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            patience_ctr = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1}, ckpt_path)
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}: train={train_losses[-1]:.4f}, val={vl:.4f}")
        if patience_ctr >= C.PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False)['model'])

    model.eval()
    with torch.no_grad():
        Y_pred, _, _, _, gate_weights = model(Xt_t, fft_t_t, bond_t_t, deriv_t_t, nmf_t_t)
        Y_pred = Y_pred.numpy()
        gate_weights = gate_weights.numpy()

    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, D)
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0

    # Spoke analysis
    spoke_names = ['VAE', 'PCA', 'FFT', 'BondStats', 'Derivatives', 'NMF', 'Moments', 'XCorr']
    mean_gates = gate_weights.mean(axis=0)
    metrics["spoke_gate_weights"] = {n: float(g) for n, g in zip(spoke_names, mean_gates)}
    metrics["novel_chains"] = [n for n, g in zip(spoke_names, mean_gates) if g > 0.5]

    log.info(f"M10: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
    log.info(f"M10: Spoke weights: {dict(zip(spoke_names, [f'{g:.3f}' for g in mean_gates]))}")

    # Plots
    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_radial_diagram(gate_weights, save_dir, MODEL_NUM)
    save_results(metrics, MODEL_NUM, save_dir)

    avg_chem = post_process_advanced_model(Xt, wavenumbers, Y_pred, sid_test, MODEL_NUM, save_dir)
    metrics["chemistry"] = avg_chem

    gc.collect()
    return Y_pred, metrics
