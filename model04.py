"""
Model 04: MLP (Multi-Layer Perceptron) — Deep learning baseline.
1024→256→BN→ReLU→Drop→128→BN→ReLU→Drop→64→ReLU→6→Softmax
Loss: KL Divergence. Optimizer: AdamW. Scheduler: CosineAnnealingWarmRestarts.
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

log = logging.getLogger(__name__)
MODEL_NUM = 4


class MLPModel(nn.Module):
    def __init__(self, in_dim=1024, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Linear(64, C.NUM_AA),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


def augment_batch(X, Y, n_gauss=C.N_GAUSSIAN, n_mix=C.N_MIXUP, noise_std=C.NOISE_STD):
    """Gaussian noise + Mixup augmentation."""
    aug_X, aug_Y = [X], [Y]

    # Gaussian noise
    for _ in range(n_gauss):
        noise = torch.randn_like(X) * noise_std
        aug_X.append(X + noise)
        aug_Y.append(Y)

    # Mixup
    for _ in range(n_mix):
        idx = torch.randperm(X.size(0))
        lam = torch.rand(X.size(0), 1).to(X.device)
        aug_X.append(lam * X + (1 - lam) * X[idx])
        aug_Y.append(lam * Y + (1 - lam) * Y[idx])

    return torch.cat(aug_X, 0), torch.cat(aug_Y, 0)


def kl_loss(pred, target):
    """KL divergence loss for compositional data."""
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

    log.info("M04: Preprocessing...")
    Xtr = preprocess_batch(X_train, 'full')
    Xv = preprocess_batch(X_val, 'full')
    Xt = preprocess_batch(X_test, 'full')

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_train, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    Yv_t = torch.tensor(Y_val, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)

    model = MLPModel(in_dim=Xtr.shape[1], drop=C.DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Check for checkpoint
    start_epoch = 0
    if os.path.exists(ckpt_path) and not retrain:
        log.info("M04: Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    train_ds = TensorDataset(Xtr_t, Ytr_t)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)

    log.info(f"M04: Training from epoch {start_epoch}...")
    for epoch in range(start_epoch, C.MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for xb, yb in train_loader:
            xb_aug, yb_aug = augment_batch(xb, yb)
            pred = model(xb_aug)
            loss = kl_loss(pred, yb_aug)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t)
            val_loss = kl_loss(val_pred, Yv_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1}, ckpt_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if patience_counter >= C.PATIENCE:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])

    # Predict
    model.eval()
    with torch.no_grad():
        Y_pred = model(Xt_t).numpy()

    Y_pred = softmax_normalize(Y_pred)

    metrics = compute_metrics(Y_test, Y_pred, Xtr.shape[1])
    metrics["epochs"] = len(train_losses)
    metrics["training_time"] = time.time() - t0

    log.info(f"M04: Test R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")

    plot_loss_curve(train_losses, val_losses, MODEL_NUM, save_dir)
    plot_scatter_aggregated(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    plot_scatter_raw(Y_test, Y_pred, sid_test, MODEL_NUM, save_dir)
    save_results(metrics, MODEL_NUM, save_dir)

    gc.collect()
    return Y_pred, metrics
