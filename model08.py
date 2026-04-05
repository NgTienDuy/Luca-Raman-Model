"""
Model 08: Regularization & Multi-task Optimizer.
Jointly predict composition + physicochemical properties.
NMF-like unmixing module. Multi-regularization experiments.
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
from chemistry_report import extract_chemistry
from model_helpers import post_process_advanced_model

log = logging.getLogger(__name__)
MODEL_NUM = 8


class MultiTaskModel(nn.Module):
    def __init__(self, in_dim, n_regions, n_components=6, drop=0.3):
        super().__init__()
        self.n_components = n_components

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        # NMF-like unmixing: learnable pure component matrix
        self.pure_components = nn.Parameter(torch.randn(n_components, in_dim) * 0.01)
        self.concentration_head = nn.Linear(128, n_components)

        # Reconstruction from concentrations
        # concentration @ pure_components → reconstructed spectrum

        # Composition head
        self.comp_head = nn.Sequential(
            nn.Linear(128 + n_components, 64),
            nn.ReLU(),
            nn.Linear(64, C.NUM_AA),
        )

        # Chemistry heads (self-supervised)
        self.chem_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # pH_score, polarity, hydrophilicity, bond_strength
        )

    def forward(self, x):
        feat = self.encoder(x)

        # NMF-like concentrations (non-negative)
        conc = F.softplus(self.concentration_head(feat))  # (B, K)
        conc_norm = conc / (conc.sum(dim=-1, keepdim=True) + 1e-8)

        # Reconstruction
        pure = F.softplus(self.pure_components)  # non-negative
        recon = conc @ pure  # (B, in_dim)

        # Composition prediction
        comp_input = torch.cat([feat, conc_norm], dim=-1)
        comp = F.softmax(self.comp_head(comp_input), dim=-1)

        # Chemistry prediction
        chem = self.chem_head(feat)

        return comp, recon, chem, conc_norm


def kl_loss(pred, target):
    pred = torch.clamp(pred, 1e-8, 1.0)
    target = torch.clamp(target, 1e-8, 1.0)
    return F.kl_div(pred.log(), target, reduction='batchmean')


def _compute_chem_targets(X, wavenumbers, Y):
    """Compute chemistry targets for self-supervised learning."""
    targets = np.zeros((len(X), 4), dtype=np.float32)
    for i in range(len(X)):
        props = extract_chemistry(X[i], wavenumbers, Y[i])
        targets[i, 0] = props.get("pH_score", 0)
        targets[i, 1] = props.get("polarity", 0)
        targets[i, 2] = props.get("hydrophilicity", 0)
        targets[i, 3] = props.get("bond_strength", 0)
    # Normalize
    m, s = targets.mean(0), targets.std(0) + 1e-8
    return (targets - m) / s, m, s


def run(X_train, X_val, X_test, Y_train, Y_val, Y_test,
        wavenumbers=None, retrain=False, **kw):
    set_seeds()
    t0 = time.time()
    save_dir = os.path.join(C.RESULTS_DIR, f"model{MODEL_NUM:02d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(C.CHECKPOINTS_DIR, f"model{MODEL_NUM:02d}.pt")
    sid_test = kw.get("sid_test")

    log.info("M08: Preprocessing...")
    Xtr = preprocess_batch(X_train, 'sg_snv')
    Xv = preprocess_batch(X_val, 'sg_snv')
    Xt = preprocess_batch(X_test, 'sg_snv')

    log.info("M08: Computing chemistry targets...")
    chem_tr, chem_m, chem_s = _compute_chem_targets(Xtr, wavenumbers, Y_train)

    D = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)
    chem_tr_t = torch.tensor(chem_tr, dtype=torch.float32)

    model = MultiTaskModel(D, N_REGIONS, n_components=6, drop=C.DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    ds = TensorDataset(Xtr_t, Ytr_t, chem_tr_t)
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True)

    train_losses, val_losses = [], []
    best_val = float('inf')
    patience_ctr = 0

    log.info("M08: Training...")
    for epoch in range(C.MAX_EPOCHS):
        model.train()
        ep_loss = 0
        nb = 0
        for xb, yb, cb in loader:
            comp, recon, chem, conc = model(xb)
            l_kl = kl_loss(comp, yb)
            l_recon = F.mse_loss(recon, xb)
            l_chem = F.mse_loss(chem, cb)

            # L1 on concentrations (sparsity)
            l_sparse = torch.mean(torch.abs(conc))

            loss = l_kl + C.LAMBDA_RECON * l_recon + C.LAMBDA_CHEM * l_chem + 0.001 * l_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            vp, _, _, _ = model(Xv_t)
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
        Y_pred, _, _, _ = model(Xt_t)
        Y_pred = Y_pred.numpy()
    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, D)
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0

    log.info(f"M08: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    avg_chem = post_process_advanced_model(Xt, wavenumbers, Y_pred, sid_test, MODEL_NUM, save_dir)
    metrics["chemistry"] = avg_chem

    gc.collect()
    return Y_pred, metrics
