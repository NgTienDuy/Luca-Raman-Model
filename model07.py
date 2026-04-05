"""
Model 07: Spectral Feature Extraction Optimizer.
Multi-head attention over bond regions. Dictionary learning approach.
Multi-task loss: L_KL + λ_pos·L_pos + λ_int·L_int + λ_bond·L_bondtype + λ_attn·H(A)
"""
import os, time, logging, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config as C
from utils import (preprocess_batch, compute_metrics, softmax_normalize, plot_scatter_aggregated, plot_scatter_raw,
                   plot_loss_curve, save_results, set_seeds)
from spectral_knowledge import REGIONS, N_REGIONS, extract_bond_features
from model_helpers import post_process_advanced_model

log = logging.getLogger(__name__)
MODEL_NUM = 7


class BondAttentionNet(nn.Module):
    def __init__(self, in_dim, n_regions, drop=0.3):
        super().__init__()
        self.n_regions = n_regions
        bond_feat_dim = n_regions * 5  # 5 features per region
        deriv_dim = 16  # 8 segments * 2

        # Bond feature encoder
        self.bond_enc = nn.Sequential(
            nn.Linear(bond_feat_dim + deriv_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop * 0.5),
        )

        # Spectral encoder
        self.spec_enc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Multi-head attention over regions
        self.attn_q = nn.Linear(128, n_regions)
        self.attn_k = nn.Linear(128, n_regions)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.head = nn.Linear(64, C.NUM_AA)

        # Auxiliary heads for multi-task
        self.peak_pos_head = nn.Linear(64, n_regions)  # predicted peak positions
        self.peak_int_head = nn.Linear(64, n_regions)  # predicted intensities

    def forward(self, x_spec, x_bond):
        spec_feat = self.spec_enc(x_spec)  # (B, 128)
        bond_feat = self.bond_enc(x_bond)  # (B, 128)

        # Attention
        q = self.attn_q(spec_feat)  # (B, n_regions)
        k = self.attn_k(bond_feat)  # (B, n_regions)
        attn = F.softmax(q * k / (self.n_regions ** 0.5), dim=-1)  # (B, n_regions)

        # Weight bond features by attention
        bond_weighted = bond_feat * attn.sum(dim=-1, keepdim=True)

        fused = torch.cat([spec_feat, bond_weighted], dim=-1)
        feat = self.fusion(fused)

        pred = F.softmax(self.head(feat), dim=-1)
        peak_pos = self.peak_pos_head(feat)
        peak_int = self.peak_int_head(feat)

        return pred, peak_pos, peak_int, attn


def kl_loss(pred, target):
    pred = torch.clamp(pred, 1e-8, 1.0)
    target = torch.clamp(target, 1e-8, 1.0)
    return F.kl_div(pred.log(), target, reduction='batchmean')


def entropy_loss(attn):
    """Encourage attention entropy (spread)."""
    attn = torch.clamp(attn, 1e-8, 1.0)
    return -torch.mean(torch.sum(attn * attn.log(), dim=-1))


def _extract_bond_batch(X, wavenumbers):
    """Extract bond + derivative features for batch."""
    from spectral_knowledge import extract_bond_features, extract_derivative_features
    feats = []
    for i in range(len(X)):
        bf = extract_bond_features(X[i], wavenumbers, REGIONS).flatten()
        df = extract_derivative_features(X[i], wavenumbers)
        feats.append(np.concatenate([bf, df]))
    return np.array(feats, dtype=np.float32)


def _extract_targets(X, wavenumbers):
    """Extract peak positions and intensities as auxiliary targets."""
    n = len(X)
    pos = np.zeros((n, N_REGIONS), dtype=np.float32)
    ints = np.zeros((n, N_REGIONS), dtype=np.float32)
    for i in range(n):
        bf = extract_bond_features(X[i], wavenumbers, REGIONS)
        pos[i] = bf[:, 2]  # peak_position
        ints[i] = bf[:, 1]  # peak_height
    return pos, ints


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINTS_DIR, f"model{MODEL_NUM:02d}.pt")
    sid_test = kw.get("sid_test")

    log.info("M07: Preprocessing and feature extraction...")
    Xtr = preprocess_batch(X_train, 'sg_snv')
    Xv = preprocess_batch(X_val, 'sg_snv')
    Xt = preprocess_batch(X_test, 'sg_snv')

    bf_tr = _extract_bond_batch(Xtr, wavenumbers)
    bf_v = _extract_bond_batch(Xv, wavenumbers)
    bf_t = _extract_bond_batch(Xt, wavenumbers)

    pos_tr, int_tr = _extract_targets(Xtr, wavenumbers)
    pos_v, int_v = _extract_targets(Xv, wavenumbers)

    # Normalize targets
    pos_mean, pos_std = pos_tr.mean(), pos_tr.std() + 1e-8
    int_mean, int_std = int_tr.mean(), int_tr.std() + 1e-8
    pos_tr_n = (pos_tr - pos_mean) / pos_std
    int_tr_n = (int_tr - int_mean) / int_std

    D = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    bf_tr_t = torch.tensor(bf_tr, dtype=torch.float32)
    bf_v_t = torch.tensor(bf_v, dtype=torch.float32)
    bf_t_t = torch.tensor(bf_t, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)
    pos_tr_t = torch.tensor(pos_tr_n, dtype=torch.float32)
    int_tr_t = torch.tensor(int_tr_n, dtype=torch.float32)

    model = BondAttentionNet(D, N_REGIONS, C.DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    ds = TensorDataset(Xtr_t, bf_tr_t, Ytr_t, pos_tr_t, int_tr_t)
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True)

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_ctr = 0

    log.info("M07: Training...")
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        ep_loss = 0
        nb = 0
        for xs, xb, yb, pb, ib in loader:
            pred, pp, pi, attn = model(xs, xb)
            l_kl = kl_loss(pred, yb)
            l_pos = F.mse_loss(pp, pb)
            l_int = F.mse_loss(pi, ib)
            l_ent = entropy_loss(attn)
            loss = l_kl + C.LAMBDA_POS * l_pos + C.LAMBDA_INT * l_int + C.LAMBDA_ATTN * l_ent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vp, _, _, _ = model(Xv_t, bf_v_t)
            vl = kl_loss(vp, Yv_t).item()
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
        Y_pred, _, _, _ = model(Xt_t, bf_t_t)
        Y_pred = Y_pred.numpy()
    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, D)
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0

    log.info(f"M07: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    avg_chem = post_process_advanced_model(Xt, wavenumbers, Y_pred, sid_test, MODEL_NUM, save_dir)
    metrics["chemistry"] = avg_chem

    gc.collect()
    return Y_pred, metrics
