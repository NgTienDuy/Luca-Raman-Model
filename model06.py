"""
Model 06: Adaptive Preprocessing Optimizer.
Gate network selects optimal preprocessing per-spectrum.
Multi-task loss: L_KL + λ_recon·MSE + λ_smooth·TV
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
from model_helpers import post_process_advanced_model

log = logging.getLogger(__name__)
MODEL_NUM = 6


class AdaptivePreproc(nn.Module):
    def __init__(self, in_dim, n_variants=4, drop=0.3):
        super().__init__()
        self.n_variants = n_variants

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_variants),
        )

        # Main predictor (same as M4 backbone)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, C.NUM_AA)
        self.recon_head = nn.Linear(64, in_dim)  # for reconstruction loss

    def forward(self, x_variants, x_raw):
        """
        x_variants: (B, K, D) — K preprocessing variants
        x_raw: (B, D) — raw spectrum
        """
        gate_w = F.softmax(self.gate(x_raw), dim=-1)  # (B, K)
        # Weighted combination
        x_fused = torch.sum(gate_w.unsqueeze(-1) * x_variants, dim=1)  # (B, D)

        feat = self.backbone(x_fused)
        pred = F.softmax(self.head(feat), dim=-1)
        recon = self.recon_head(feat)
        return pred, recon, gate_w, x_fused


def total_variation(x):
    """Total variation smoothness penalty."""
    return torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))


def kl_loss(pred, target):
    pred = torch.clamp(pred, 1e-8, 1.0)
    target = torch.clamp(target, 1e-8, 1.0)
    return F.kl_div(pred.log(), target, reduction='batchmean')


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINTS_DIR, f"model{MODEL_NUM:02d}.pt")
    sid_test = kw.get("sid_test")

    log.info("M06: Preprocessing variants...")
    variants = ['none', 'snv', 'sg_snv']  # reduced from 4 to 3 for memory
    K = len(variants)

    # Preprocess all variants sequentially to save memory
    Xtr_list = []
    Xv_list = []
    Xt_list = []
    for m in variants:
        Xtr_list.append(preprocess_batch(X_train, m))
        Xv_list.append(preprocess_batch(X_val, m))
        Xt_list.append(preprocess_batch(X_test, m))
    Xtr_vars = np.stack(Xtr_list, axis=1); del Xtr_list
    Xv_vars = np.stack(Xv_list, axis=1); del Xv_list
    Xt_vars = np.stack(Xt_list, axis=1); del Xt_list
    gc.collect()

    Xtr_raw = preprocess_batch(X_train, 'none')
    Xv_raw = preprocess_batch(X_val, 'none')
    Xt_raw = preprocess_batch(X_test, 'none')
    Xtr_full = preprocess_batch(X_train, 'sg_snv')

    D = Xtr_raw.shape[1]

    Xtr_vars_t = torch.tensor(Xtr_vars, dtype=torch.float32)
    Xv_vars_t = torch.tensor(Xv_vars, dtype=torch.float32)
    Xt_vars_t = torch.tensor(Xt_vars, dtype=torch.float32)
    Xtr_raw_t = torch.tensor(Xtr_raw, dtype=torch.float32)
    Xv_raw_t = torch.tensor(Xv_raw, dtype=torch.float32)
    Xt_raw_t = torch.tensor(Xt_raw, dtype=torch.float32)
    Xtr_full_t = torch.tensor(Xtr_full, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)

    model = AdaptivePreproc(D, K, C.DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_ctr = 0

    ds = TensorDataset(Xtr_vars_t, Xtr_raw_t, Xtr_full_t, Ytr_t)
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True)

    log.info("M06: Training...")
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        ep_loss = 0
        nb = 0
        for xv_b, xr_b, xf_b, y_b in loader:
            pred, recon, gw, x_fused = model(xv_b, xr_b)
            l_kl = kl_loss(pred, y_b)
            l_recon = F.mse_loss(recon, xf_b)
            l_tv = total_variation(x_fused)
            loss = l_kl + C.LAMBDA_RECON * l_recon + C.LAMBDA_SMOOTH * l_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vp, _, _, _ = model(Xv_vars_t, Xv_raw_t)
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
        Y_pred, _, _, _ = model(Xt_vars_t, Xt_raw_t)
        Y_pred = Y_pred.numpy()
    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, D)
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0

    log.info(f"M06: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    Xt_full = preprocess_batch(X_test, 'sg_snv')
    avg_chem = post_process_advanced_model(Xt_full, wavenumbers, Y_pred, sid_test, MODEL_NUM, save_dir)
    metrics["chemistry"] = avg_chem

    gc.collect()
    return Y_pred, metrics
